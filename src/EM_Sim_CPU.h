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

    void get_PML_info(float kappas[DIRECTIONS], float Qs[DIRECTIONS], int i, int j);
    void step_EM(int step_index);
};

void EM_Sim_CPU::get_PML_info(float kappas[DIRECTIONS], float Qs[DIRECTIONS], int i, int j)
{
    PML *pmls[DIRECTIONS] = {pml_xdir, pml_ydir};
    int N_along_dir[DIRECTIONS] = {Nx, Ny}; 
    int index_along_dir[DIRECTIONS] = {i, j};
    // float Ez_diff[DIRECTIONS] = {
    //     Ez[k_for_ij] - Ez[k_for_ip1j], 
    //     Ez[k_for_ij] - Ez[k_for_ijp1] };
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
        if (pml_node != NULL)
        { 
            kappa = pml_node->kappa_M;
            Q = pml_node->Q;
            b = pml_node->b_M;
            c = pml_node->c_M;
            kappas[d] = kappa;
            //Qs[d] = b*Q - c*Ez_diff[d];
            Qs[d] = b*Q;
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
                    float Qs[DIRECTIONS] = {0, 0};
                    get_PML_info(kappas, Qs, i, j);
                    float Ez_diff[DIRECTIONS] = {
                        Ez[k_for_ij] - Ez[k_for_ip1j], 
                        Ez[k_for_ij] - Ez[k_for_ijp1] };
                    Hy[k_for_ij] += (coef_mu_dx / kappas[X_DIR]) * (Ez_diff[X_DIR]) + Qs[X_DIR];
                    Hx[k_for_ij] -= (coef_mu_dy / kappas[Y_DIR]) * (Ez_diff[Y_DIR]) + Qs[Y_DIR];                     
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
                Ez[k_for_ij] += coef_eps_dx*(Hy[k_for_im1j] - Hy[k_for_ij]) - coef_eps_dy*(Hx[k_for_ijm1] - Hx[k_for_ij]);
            }
            if (PRINT)
                printf("E-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }

    EM_Probe_Manager &probeManager = EM_Probe_Manager::instance();
    probeManager.probe(Ez, step_index);
}