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
        float bs[DIRECTIONS], float cs[DIRECTIONS], value_t field_diffs[DIRECTIONS], int i, int j, bool is_for_M);
    void step_EM(int step_index);
};

void EM_Sim_CPU::get_PML_info(float kappas[DIRECTIONS], 
    float bs[DIRECTIONS], float cs[DIRECTIONS], value_t field_diffs[DIRECTIONS], int i, int j, bool is_for_M)
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
        float Q = 0, b = 0, c = 0;
        value_t field_diff = field_diffs[d];
        if (pml_node != NULL)
        { 
            if (is_for_M)
            {
                kappa = pml_node->kappa_M;
                b = pml_node->b_M;
                c = pml_node->c_M;  
                //Q_M[d][k_for_ij] = b*Q_M[d][k_for_ij] - c*field_diff;
            }
            else
            {
                kappa = pml_node->kappa_E;
                b = pml_node->b_E;
                c = pml_node->c_E;   
            }
            kappas[d] = kappa;
            bs[d] = b;
            cs[d] = c;
        }
    }
}

void EM_Sim_CPU::step_EM(int step_index)
{       
    // Magnetic Field Update
    #pragma omp parallel for num_threads(num_threads) collapse(2) if(do_parallel)   
    for (int i=x_fi; i<x_li; i++)
    {        
        for (int j=y_fi; j<y_li; j++)
        {            
            int k_for_ij = ij_to_k(i, j);
            int k_for_ijp1 = ij_to_k(i, j+1);
            int k_for_ip1j = ij_to_k(i+1, j); 
            
            int material_value = 0;
            if(materialData.has_material)
                material_value = materialData.scaled_data[k_for_ij];
            
            if (material_value == 1)
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
                    float bs[DIRECTIONS] = {0, 0};
                    float cs[DIRECTIONS] = {0, 0};
                    value_t *Q_M[DIRECTIONS] = {Q_M_x, Q_M_y};
                    bool is_for_M = true; 
                    float Ez_diff[DIRECTIONS] = {
                        Ez[k_for_ij] - Ez[k_for_ip1j], 
                        Ez[k_for_ij] - Ez[k_for_ijp1] };
                    get_PML_info(kappas, bs, cs, Ez_diff, i, j, is_for_M);
                    for (int dir=0; dir<DIRECTIONS; ++dir)
                    {
                        value_t *Q = Q_M[dir];
                        Q[k_for_ij] = bs[dir]*Q[k_for_ij] - cs[dir]*Ez_diff[dir];
                    }
                    Hy[k_for_ij] += (coef_mu_dx / kappas[X_DIR]) * (Ez_diff[X_DIR]) + Q_M[X_DIR][k_for_ij];
                    Hx[k_for_ij] -= (coef_mu_dy / kappas[Y_DIR]) * (Ez_diff[Y_DIR]) + Q_M[Y_DIR][k_for_ij];                     
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
                        
            int material_value = 0;
            if(materialData.has_material)
                material_value = materialData.scaled_data[k_for_ij];            
            
            if (material_value == 1)
            {
                Ez[k_for_ij] = 0;    
            }
            else
            {
                float kappas[DIRECTIONS] = {1.0, 1.0};
                float bs[DIRECTIONS] = {0, 0};
                float cs[DIRECTIONS] = {0, 0};
                value_t *Q_E[DIRECTIONS] = {Q_E_x, Q_E_y};
                bool is_for_M = false; 
                value_t H_diff[DIRECTIONS] = {
                    Hy[k_for_im1j] - Hy[k_for_ij], 
                    Hx[k_for_ijm1] - Hx[k_for_ij] };
                get_PML_info(kappas, bs, cs, H_diff, i, j, is_for_M); 
                for (int dir=0; dir<DIRECTIONS; ++dir)
                {
                    value_t *Q = Q_E[dir];
                    Q[k_for_ij] = bs[dir]*Q[k_for_ij] - cs[dir]*H_diff[dir];
                } 
                Ez[k_for_ij] += (coef_eps_dx / kappas[X_DIR])*(Hy[k_for_im1j] - Hy[k_for_ij]) -
                                (coef_eps_dy / kappas[Y_DIR])*(Hx[k_for_ijm1] - Hx[k_for_ij]) +
                                + Q_E[X_DIR][k_for_ij] + Q_E[Y_DIR][k_for_ij];
            }
            if (PRINT)
                printf("E-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }

    EM_Probe_Manager &probeManager = EM_Probe_Manager::instance();
    probeManager.probe(Ez, step_index);
}