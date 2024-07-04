#include <omp.h>
#include "base.h"
#include "Material.h"
#include "EM_Sim.h"
#include "Config.h"

#define PRINT 0

int do_parallel = true;

class EM_Probe
{
private:
    std::string name;
public:
    int total_steps;
    value_t *values;

    EM_Probe(int total_steps_)
    {
        total_steps = total_steps_;
        values = new value_t[total_steps];
        initialize_zero(values, total_steps);
    }

    void save()
    {
        FileManager &fileManager = FileManager::instance();    
        auto path = fileManager.into_data_dir("em_prob.json");
        json j;
        j["name"] = name;
        std::vector<value_t> vec_values(values, values+total_steps);
        j["values"] = vec_values;
        fileManager.save_json(j, path);            
    }

    ~EM_Probe()
    {
        delete values;
    }
};

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
    EM_Sim_CPU(value_t *Hx, value_t *Hy, value_t *Ez, MaterialData &material_data) 
    : EM_Sim(Hx, Hy, Ez, material_data)
    {
        num_threads = Config::instance().num_threads;
    }

    void step_EM();
};

void EM_Sim_CPU::step_EM()
{       
    // Magnetic Field Update
    #pragma omp parallel for num_threads(num_threads) collapse(2) if(do_parallel)   
    for (int i=x_fi; i<x_li; i++)
    {        
        for (int j=y_fi; j<y_li; j++)
        {
            
            int k_for_ij = ij_to_k(i, j, Nx);
            int k_for_ijp1 = ij_to_k(i, j+1, Nx);
            int k_for_ip1j = ij_to_k(i+1, j, Nx); 
            int material_value = materialData.scaled_data[k_for_ij];
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
    #pragma omp parallel for num_threads(num_threads) collapse(2) if(do_parallel)
    for (int i=(x_fi+1); i<x_li; i++)
    {
        for (int j=(y_fi+1); j<y_li; j++)
        {
            int k_for_ij = ij_to_k(i, j, Nx);
            int k_for_ijm1 = ij_to_k(i, j-1, Nx);
            int k_for_im1j = ij_to_k(i-1, j, Nx); 
            
            int material_value = materialData.scaled_data[k_for_ij];
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