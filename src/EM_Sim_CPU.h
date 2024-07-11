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

    void step_EM(int step_index);
};

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
                    PML_Node *pml_node_xdir = NULL;
                    PML_Node *pml_node_ydir = NULL;
                    int n_PML_xdir = pml_xdir->n_PML_nodes_per_part; 
                    int n_PML_ydir = pml_ydir->n_PML_nodes_per_part;                    
                    pml_node_xdir = get_PML_node(pml_xdir->part1,
                        pml_xdir->part2, n_PML_xdir, Nx, i);
                    pml_node_ydir = get_PML_node(pml_ydir->part1,
                        pml_ydir->part2, n_PML_ydir, Ny, j);                    
                    
                    float kappa_x = 1.0, kappa_y = 1.0;                    
                    float Q_x = 0, b_x = 0, c_x = 0;
                    float Q_y = 0, b_y = 0, c_y = 0;
                    if (pml_node_xdir != NULL)
                    { 
                        kappa_x = pml_node_xdir->kappa_M;
                        Q_x = pml_node_xdir->Q;
                        b_x = pml_node_xdir->b_M;
                        c_x = pml_node_xdir->c_M;
                    }
                    if (pml_node_ydir != NULL)
                    { 
                        kappa_y = pml_node_ydir->kappa_M;
                        Q_y = pml_node_ydir->Q;
                        b_y = pml_node_ydir->b_M;
                        c_y = pml_node_ydir->c_M;
                    }
                    Hx[k_for_ij] -= (coef_mu_dy / kappa_y) * (Ez[k_for_ij] - Ez[k_for_ijp1]) + Q_y;                     
                    Hy[k_for_ij] += (coef_mu_dx / kappa_x) * (Ez[k_for_ij] - Ez[k_for_ip1j]) + Q_x;
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