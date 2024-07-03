#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono> 
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <filesystem>
#include <cassert>
#include <cstdlib>
#include "base.h"
#include "Material.h"
#include "FileManager.h"
#include "Timer.h"
#include "EM_Sim.h"

#ifdef USE_GPU
    #include "step_EM_gpu.h"
#else
    #include "EM_Sim_CPU.h"
#endif        



bool do_logging = true;

int main()
{            
    FileManager &fileManager = FileManager::instance();    
    fileManager.init();
    
    Material material("data/car_interior_2D_image_data.json");    
    material.parse();
    MaterialData material_data = material.scaleToFit(Nx, Ny);            
    
    printf("c0: %f, Nx: %d, Ny:%d, L0: %f, dx: %f, dt: %.9f, space_x: %f,space_y: %f\n", c0, Nx, Ny, lam, dx, dt, space_size_x, space_size_y);
    
    int N = Nx * Ny;
    value_t *Ez = new value_t[N];
    value_t *Hx = new value_t[N];
    value_t *Hy = new value_t[N];
    
    // Initialize arrays
    #pragma omp parallel for num_threads(NUM_THREADS) if(do_parallel)
    for (int i=0; i<N; i++)
    {
        Ez[i] = 0;
        Hx[i] = 0;
        Hy[i] = 0;
    }

    // write scaled material data
    Timer timer;
    std::vector<int> material_values(material_data.scaled_data, material_data.scaled_data+N);
    int vec_size = material_values.size();
    json j;
    j["material_data_size"] = vec_size;
    j["material_data"] = material_values;
    j["Nx"] = Nx;
    j["Ny"] = Ny;
    std::string path = fileManager.convert_to_path("material.json");
    fileManager.save_json(j, path);    
    timer.end();
    timer.print_elapsed_time("<material.json> save elapsed time");
    assert (vec_size == N);    

    int logging_period = 5;
    int test = 0;

    //std::vector<float> vec_Ez(N);    
    json j_sim;
    j_sim["Nx"] = Nx;
    j_sim["Ny"] = Ny;
    j_sim["N"] = N;  
    j_sim["logging_period"] = logging_period;              
    
    EM_Sim_CPU sim_cpu(Hx, Hy, Ez, material_data);
    EM_Sim *sim = (EM_Sim *)(&sim_cpu);

    //output_file << Nx << "," << Ny << "," << nt << "," << logging_period << "\n";
    float time_for_data_write = 0;
    int computation_time = 0; 
    for (int step=0; step<nt; step++)
    {        
        //Point Source        
        int source_k = ij_to_k((int)(Nx*0.15), (int)(Ny*0.7), Nx);
        Ez[source_k] += sin(2*M_PI*f0*(dt*step)) * exp(-0.5*pow((step-20)/8, 2));
        
        timer.begin();       
        sim->step_EM();        
        float elapsed_time_micro = timer.end(false);                         
        computation_time += elapsed_time_micro;
        
        // logging
        if (do_logging && step % logging_period == 0)
        {
            timer.begin();
            //vec_Ez.assign(Ez, Ez+N);
            std::vector<float> vec_Ez(Ez, Ez + N);
            std::string time_stamp = fmt::format("t{}", step);
            j_sim[time_stamp] = vec_Ez;
            float elapsed_time = timer.end();
            time_for_data_write += elapsed_time;             
        }     
    }
    computation_time /= 1000.0; // to ms
    path = fileManager.convert_to_path("output_cpu.json");
    fileManager.save_json(j_sim, path);    
        
    std::cout << "EM computation time: " << computation_time << " ms" << std::endl;
    std::cout << "data write time: " << time_for_data_write << " ms" << std::endl;
}
