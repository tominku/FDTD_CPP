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

#include "nlohmann/json.hpp"
using json = nlohmann::json;

using namespace std;
using namespace std::chrono;

#include "step_EM_cpu.h"

bool do_logging = true;

int main()
{    
    FileManager &fileManager = FileManager::instance();    
    fileManager.init();
    
    Material material("data/car_interior_2D_image_data.json");    
    material.parse();
    MaterialData material_data = material.scaleToFit(Nx, Ny);            

    // Define Simulation Based off Source and Wavelength
    int f0 = 1e6; // Frequency of Source  [Hertz]
    int nt = 2000; // Number of time steps  [unitless]

    // Spatial and Temporal System
    value_t eps0 = 8.854 * 1e-12;  // Permittivity of vacuum [farad/meter]
    value_t mu0 = 4*M_PI* 1e-7;  // Permeability of vacuum [henry/meter]
    value_t c0 = 1/pow((eps0*mu0), 0.5);  // Speed of light  [meter/second]
    value_t lam = c0/f0;  // Freespace Wavelength  [meter]
    value_t t0  = 1/f0;  // Source Period  [second]

    value_t space_size_x = num_waves_x * lam;
    value_t space_size_y = num_waves_y * lam;
    value_t dx = space_size_x / (Nx-1);
    value_t dy = space_size_y / (Ny-1);
    value_t dt = pow(pow(dx,-2) + pow(dy,-2), -0.5)/c0*.99;

    value_t coef_eps_dx = dt/(eps0*dx);
    value_t coef_eps_dy = dt/(eps0*dy);
    value_t coef_mu_dx = dt/(mu0*dx);
    value_t coef_mu_dy = dt/(mu0*dy);
    
    /*
    [Nx,Ny] = deal(Lx*Lf,Ly*Lf);    % Points in x,y           [unitless]
    x  = linspace(0,Lx,Nx+1)*L0;    % x vector                [meter]
    y  = linspace(0,Ly,Ny+1)*L0;    % y vector                [meter]
    [dx,dy] = deal(x(2),y(2));      % x,y,z increment         [meter]
    dt = (dx^-2+dy^-2)^-.5/c0*.99;  % Time step CFL condition [second]
    */
    
    printf("c0: %f, Nx: %d, Ny:%d, L0: %f, dx: %f, dt: %.9f, space_x: %f,space_y: %f\n", c0, Nx, Ny, lam, dx, dt, space_size_x, space_size_y);

    int computation_time = 0; 
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
    
    //output_file << Nx << "," << Ny << "," << nt << "," << logging_period << "\n";
    float time_for_data_write = 0;
    for (int step=0; step<nt; step++)
    {        
        //Point Source        
        int source_k = ij_to_k((int)(Nx*0.15), (int)(Ny*0.7), Nx);
        Ez[source_k] += sin(2*M_PI*f0*(dt*step)) * exp(-0.5*pow((step-20)/8, 2));
        
        auto t1 = steady_clock::now();
        
        step_em_pml(Hx, Hy, Ez, coef_eps_dx, coef_eps_dy, coef_mu_dx, coef_mu_dy, material_data);        
        
        auto t2 = steady_clock::now();

        auto duration = duration_cast<microseconds>(t2 - t1);
        computation_time += duration.count();
        
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
    path = fileManager.convert_to_path("output_cpu.json");
    fileManager.save_json(j_sim, path);    
    
    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << "computation time: " << computation_time / 1000 << " ms" << std::endl;
    std::cout << "data write time: " << time_for_data_write << " ms" << std::endl;
}
