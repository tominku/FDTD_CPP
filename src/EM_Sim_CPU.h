#include <omp.h>
#include "base.h"
#include "Material.h"
#include "EM_Sim.h"
#include "Config.h"
#include "EM_Probe_Manager.h"

#define PRINT 0

int do_parallel = true;

class EM_Sim_CPU : EM_Sim
{
private:
    int num_threads;    

protected:
    std::string toName()
    {
        return "EM_Sim_CPU";
    }

public:
    EM_Sim_CPU(value_t *Hx, value_t *Hy, value_t *Ez, MaterialData &material_data, bool use_pml_) 
    : EM_Sim(Hx, Hy, Ez, material_data, use_pml_)
    {
        num_threads = Config::instance().num_threads;        
    }

    void get_PML_info(float kappas[DIRECTIONS], 
        float bs[DIRECTIONS], float cs[DIRECTIONS],
        value_t field_diffs[DIRECTIONS], int i, int j, bool is_for_M,
        float sigmas[DIRECTIONS]);
    void step_EM(int step_index);
};

void EM_Sim_CPU::get_PML_info(float kappas[DIRECTIONS], 
    float bs[DIRECTIONS], float cs[DIRECTIONS], value_t field_diffs[DIRECTIONS], int i, int j, bool is_for_M, float sigmas[DIRECTIONS])
{
    PML *pmls[DIRECTIONS] = {pml_xdir, pml_ydir};
    int N_along_dir[DIRECTIONS] = {Nx, Ny}; 
    int index_along_dir[DIRECTIONS] = {i, j};
    value_t *Q_M[DIRECTIONS] = {Q_M_x, Q_M_y};
    value_t *Q_E[DIRECTIONS] = {Q_E_x, Q_E_y};
    int k_for_ij = ij_to_k(i, j);
    for (int d=0; d<DIRECTIONS; ++d)
    {   
        PML *pml = pmls[d];
        PML_Node *pml_node = NULL;
        int n_PML = pml->n_PML_nodes_per_part;
        pml_node = get_PML_node(
            pml->part1, pml->part2, n_PML, 
            N_along_dir[d], index_along_dir[d]);
        float kappa = 1.0;   
        float sigma = 0.9;                 
        float Q = 0, b = 0, c = 0;
        value_t field_diff = field_diffs[d];
        if (pml_node != NULL)
        { 
            if (is_for_M)
            {
                kappa = pml_node->kappa_M;
                b = pml_node->b_M;
                c = pml_node->c_M;  
                sigma = pml_node->sigma_M;
            }
            else
            {
                kappa = pml_node->kappa_E;
                b = pml_node->b_E;
                c = pml_node->c_E;   
                sigma = pml_node->sigma_E;
            }
            kappas[d] = kappa;
            bs[d] = b;
            cs[d] = c;
            sigmas[d] = sigma;
        }
    }
}

void EM_Sim_CPU::step_EM(int step_index)
{       
    EM_Probe_Manager &probeManager = EM_Probe_Manager::instance();
    Config &config = Config::instance();    
    // Magnetic Field Update
    #pragma omp parallel for num_threads(num_threads) collapse(2) if(do_parallel)   
    for (int i=x_fi; i<x_li; i++)
    {        
        for (int j=y_fi; j<y_li; j++)
        {            
            int k_for_ij = ij_to_k(i, j);
            int k_for_ijp1 = ij_to_k(i, j+1);
            int k_for_ip1j = ij_to_k(i+1, j); 
            
            int material_value = 1.0;
            if(materialData.has_material)
                material_value = materialData.scaled_data[k_for_ij];
            
            //if (material_value == 1)
            if (material_value < 0)
            {
                Hx[k_for_ij] = 0;    
                Hy[k_for_ij] = 0;
            }
            else
            {
                if (!use_pml)
                {
                    Hx[k_for_ij] -= coef_mu_dy * (Ez[k_for_ij] - Ez[k_for_ijp1]); 
                    Hy[k_for_ij] += coef_mu_dx * (Ez[k_for_ij] - Ez[k_for_ip1j]);
                }
                else
                {                        
                    float kappas[DIRECTIONS] = {1.0, 1.0};
                    float sigmas[DIRECTIONS] = {0, 0};
                    value_t *b_images[DIRECTIONS] = {b_x_image, b_y_image};
                    value_t *c_images[DIRECTIONS] = {c_x_image, c_y_image};
                    value_t *kappa_images[DIRECTIONS] = {kappa_x_image, kappa_y_image};
                    value_t *sigma_images[DIRECTIONS] = {sigma_x_image, sigma_y_image};
                    float bs[DIRECTIONS] = {0, 0};
                    float cs[DIRECTIONS] = {0, 0};
                    value_t *Q_M[DIRECTIONS] = {Q_M_x, Q_M_y};
                    bool is_for_M = true; 
                    float Ez_diff[DIRECTIONS] = {
                        Ez[k_for_ip1j] - Ez[k_for_ij], 
                        Ez[k_for_ijp1] - Ez[k_for_ij] };
                    float delta_spatial[DIRECTIONS] = {dx, dy};
                    get_PML_info(kappas, bs, cs, Ez_diff, i, j, is_for_M, sigmas);                    
                    for (int dir=0; dir<DIRECTIONS; ++dir)
                    {
                        value_t *Q = Q_M[dir];
                        Q[k_for_ij] = bs[dir]*Q[k_for_ij] + cs[dir]*Ez_diff[dir] / (delta_spatial[dir]);
                    }
                    float dt_over_mu = dt / mu0;
                    Hy[k_for_ij] = Hy[k_for_ij] - dt_over_mu*(-(Ez_diff[X_DIR])/(kappas[X_DIR]*delta_spatial[X_DIR]) - Q_M[X_DIR][k_for_ij]);
                    Hx[k_for_ij] = Hx[k_for_ij] - dt_over_mu*( (Ez_diff[Y_DIR])/(kappas[Y_DIR]*delta_spatial[Y_DIR]) + Q_M[Y_DIR][k_for_ij]);                 
                }
            }
            if (PRINT)
                printf("M-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
    // Electric Field Update
    #pragma omp parallel for num_threads(num_threads) collapse(2) if(do_parallel)
    for (int i=(x_fi+1); i<x_li; i++)
    {
        for (int j=(y_fi+1); j<y_li; j++)
        {
            int k_for_ij = ij_to_k(i, j);
            int k_for_ijm1 = ij_to_k(i, j-1);
            int k_for_im1j = ij_to_k(i-1, j); 
                        
            float material_value = 1.0;
            if(materialData.has_material)
                material_value = materialData.scaled_data[k_for_ij];            
            
            //if (material_value == 1)
            if (material_value < 0)
            {
                Ez[k_for_ij] = 0;    
            }
            else
            {
                float kappas[DIRECTIONS] = {1.0, 1.0};
                float sigmas[DIRECTIONS] = {0, 0};
                value_t *kappa_images[DIRECTIONS] = {kappa_x_image, kappa_y_image};
                value_t *sigma_images[DIRECTIONS] = {sigma_x_image, sigma_y_image};
                value_t *b_images[DIRECTIONS] = {b_x_image, b_y_image};
                value_t *c_images[DIRECTIONS] = {c_x_image, c_y_image};
                float bs[DIRECTIONS] = {0, 0};
                float cs[DIRECTIONS] = {0, 0};
                value_t *Q_E[DIRECTIONS] = {Q_E_x, Q_E_y};
                bool is_for_M = false; 
                value_t H_diff[DIRECTIONS] = {
                    Hy[k_for_ij] - Hy[k_for_im1j], 
                    Hx[k_for_ij] - Hx[k_for_ijm1] };
                float delta_spatial[DIRECTIONS] = {dx, dy};
                get_PML_info(kappas, bs, cs, H_diff, i, j, is_for_M, sigmas); 
                kappa_images[X_DIR][k_for_ij] = kappas[X_DIR];
                kappa_images[Y_DIR][k_for_ij] = kappas[Y_DIR];
                sigma_images[X_DIR][k_for_ij] = sigmas[X_DIR];
                sigma_images[Y_DIR][k_for_ij] = sigmas[Y_DIR];            
                for (int dir=0; dir<DIRECTIONS; ++dir)
                {
                    value_t *Q = Q_E[dir];
                    Q[k_for_ij] = bs[dir]*Q[k_for_ij] + cs[dir]*H_diff[dir] / delta_spatial[dir];
                } 
                //float dt_over_eps = dt / eps0;
                
                float J = 0;
                if (i == source_x && j == source_y)
                {
                    float f_begin = 0.6*f0; // chirp initial frequency
                    float f_end = 1.0*f0; // chirp end frequency
                    int chirp_duration_as_steps = config.total_steps; // chirp duration as steps
                    float T = chirp_duration_as_steps * dt;
                    float k = (f_end - f_begin) / T; // frequency change rate
                    //Ez[source_k] += sinf(2*M_PI*f0*(dt*step));
                    float t = dt*step_index;
                    float A = 0.1;
                    if (t <= T)
                    {
                        J = A*cosf(2*M_PI*(f_begin*t + (k/2.0)*powf(t, 2.0)) + M_PI/2);                                                            
                        probeManager.probe_Tx(J, step_index);                          
                    }
                }

                float dt_over_eps = dt / (eps0 * material_value);
                Ez[k_for_ij] = Ez[k_for_ij] + (dt_over_eps)*(
                                    (H_diff[X_DIR])/(kappas[X_DIR]*delta_spatial[X_DIR]) + Q_E[X_DIR][k_for_ij] - 
                                    (H_diff[Y_DIR])/(kappas[Y_DIR]*delta_spatial[Y_DIR]) - Q_E[Y_DIR][k_for_ij] - J );
            }
            if (PRINT)
                printf("E-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
    
    probeManager.probe(Ez, step_index);
}