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
#include "Config.h"
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
    Config &config = Config::instance();
    config.load("config_total.json");    
    
    Material &material = Material::instance();        
    MaterialData material_data = material.getMaterialData();
    
    float bw = 4*1e9; // 4 GHz; 
    float T = dt * 2000;
    float nominal_tau = T / 2;
    float S = bw / T;    
    std::string msg = fmt::format("S: {}, T: {}, c0: {}, Nx: {}, Ny: {}, L0: {}, dx: {}, dt: {}, space_x: {}, space_y: {}, source_i: {}, source_j: {}\n", S, T, c0, Nx, Ny, lam, dx, dt, space_size_x, space_size_y, source_x, source_y);
    cout << msg;
    // printf("c0: %f, Nx: %d, Ny:%d, L0: %f, dx: %f, dt: %.20f, space_x: %f,space_y: %f, source_i: %d, source_j: %d\n", c0, Nx, Ny, lam, dx, dt, space_size_x, space_size_y, source_x, source_y);
    
    int N = Nx * Ny;
    value_t *Ez = new value_t[N];
    value_t *Hx = new value_t[N];
    value_t *Hy = new value_t[N];
    
    // Initialize arrays
    initialize_zero(Hx, N);
    initialize_zero(Hy, N);
    initialize_zero(Ez, N);    
    
    Timer timer;
    // timer.end();
    // timer.print_elapsed_time("<material.json> save elapsed time");    

    int logging_period = 5;
    int test = 0;

    //std::vector<float> vec_Ez(N);    
    json j_sim;
    j_sim["Nx"] = Nx;
    j_sim["Ny"] = Ny;
    j_sim["N"] = N;  
    j_sim["logging_period"] = logging_period;              
    
    bool use_pml = true;
    //bool use_pml = false;
    EM_Sim_CPU sim_cpu(Hx, Hy, Ez, material_data, use_pml);
    EM_Sim *sim = (EM_Sim *)(&sim_cpu);
    EM_Probe_Manager &probeManager = EM_Probe_Manager::instance();
    //probeManager.add_prob_around_source(source_x + 5 , source_y + 5);

    //output_file << Nx << "," << Ny << "," << nt << "," << logging_period << "\n";
    float time_for_data_write = 0;
    int computation_time = 0; 
    int total_steps = config.total_steps;
    for (int step=0; step < total_steps; step++)
    {                
        timer.begin();       
        sim->step_EM(step);        
        float elapsed_time_micro = timer.end(false);                         
        computation_time += elapsed_time_micro;
        
        // logging
        if (do_logging && step % logging_period == 0)
        {
            timer.begin();
            //vec_Ez.assign(Ez, Ez+N);
            std::vector<value_t> vec_Ez(Ez, Ez + N);
            std::vector<value_t> vec_Hx(Hx, Hx + N);
            std::vector<value_t> vec_Hy(Hx, Hx + N);
            //std::vector<value_t> vec_kappa_x_image(sim->kappa_x_image, sim->kappa_x_image + N);
            //std::vector<value_t> vec_kappa_y_image(sim->kappa_y_image, sim->kappa_y_image + N);
            //std::vector<value_t> vec_sigma_x_image(sim->sigma_x_image, sim->sigma_x_image + N);
            //std::vector<value_t> vec_sigma_y_image(sim->sigma_y_image, sim->sigma_y_image + N);
            //std::vector<value_t> vec_b_x_image(sim->b_x_image, sim->b_x_image + N);
            //std::vector<value_t> vec_b_y_image(sim->b_y_image, sim->b_y_image + N);
            std::vector<value_t> vec_c_x_image(sim->c_x_image, sim->c_x_image + N);
            std::vector<value_t> vec_c_y_image(sim->c_y_image, sim->c_y_image + N);
            std::string time_stamp = fmt::format("t{}", step);
            //j_sim[time_stamp] = vec_c_y_image;
            //j_sim[time_stamp] = vec_c_x_image;
            //j_sim[time_stamp] = vec_sigma_y_image;
            j_sim[time_stamp] = vec_Ez;
            //j_sim[time_stamp] = vec_Hx;
            //j_sim[time_stamp] = vec_Hy;
            float elapsed_time = timer.end();
            time_for_data_write += elapsed_time;             
        }     
    }
    computation_time /= 1000.0; // to ms    
    
    // path = fileManager.into_data_dir("output_cpu.json");
    // fileManager.save_json(j_sim, path);    

    config.save_sim_frames(j_sim, "frames.json");

    probeManager.save();
        
    std::cout << "EM computation time: " << computation_time << " ms" << std::endl;
    std::cout << "data write time: " << time_for_data_write << " ms" << std::endl;
}
