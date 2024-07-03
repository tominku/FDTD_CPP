#include <omp.h>
#include "base.h"
#include "Material.h"

#define PRINT 0
#define NUM_THREADS 6

int do_parallel = true;

void step_em_pml(value_t *Hx, value_t *Hy, value_t *Ez,
    value_t coef_eps_dx, value_t coef_eps_dy, value_t coef_mu_dx, value_t coef_mu_dy, MaterialData material_data)
{   

    // Magnetic Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2) if(do_parallel)   
    for (int i=x_fi; i<x_li; i++)
    {        
        for (int j=y_fi; j<y_li; j++)
        {
            
            int k_for_ij = ij_to_k(i, j, Nx);
            int k_for_ijp1 = ij_to_k(i, j+1, Nx);
            int k_for_ip1j = ij_to_k(i+1, j, Nx); 
            int material_value = material_data.scaled_data[k_for_ij];
            //material_value = 2;
            if (material_value == 1)
            {
                Hx[k_for_ij] = 0;    
                Hy[k_for_ij] = 0;
            }
            else
            {
                Hx[k_for_ij] -= coef_mu_dy * (Ez[k_for_ij] - Ez[k_for_ijp1]); 
                Hy[k_for_ij] += coef_mu_dx * (Ez[k_for_ij] - Ez[k_for_ip1j]);
                // Hx[i][j] -= coef_mu_dy * (Ez[i][j] - Ez[i][j+1]); 
                // Hy[i][j] += coef_mu_dx * (Ez[i][j] - Ez[i+1][j]);
                }
            if (PRINT)
                printf("M-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
    // Electric Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2) if(do_parallel)
    for (int i=(x_fi+1); i<x_li; i++)
    {
        for (int j=(y_fi+1); j<y_li; j++)
        {
            int k_for_ij = ij_to_k(i, j, Nx);
            int k_for_ijm1 = ij_to_k(i, j-1, Nx);
            int k_for_im1j = ij_to_k(i-1, j, Nx); 
            
            int material_value = material_data.scaled_data[k_for_ij];
            //material_value = 2;
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
}