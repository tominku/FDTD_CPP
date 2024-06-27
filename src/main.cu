#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#define MAX_ERR 1e-6

#define devisions_per_wave 10  // Divisions per Wavelength   [unitless]
#define num_waves_x 50 //  # wave lengths in x-dir [unitless]
#define num_waves_y 50 //  # wave lengths in y-dir 
#define Nx (num_waves_x*devisions_per_wave + 1)
#define Ny (num_waves_y*devisions_per_wave + 1)
#define N (Nx*Ny)
#define Nt 1000

// Define Simulation Based off Source and Wavelength
int f0 = 1e6; // Frequency of Source  [Hertz]

// Spatial and Temporal System
double eps0 = 8.854 * 1e-12;  // Permittivity of vacuum [farad/meter]
double mu0 = 4*M_PI* 1e-7;  // Permeability of vacuum [henry/meter]
double c0 = 1/pow((eps0*mu0), 0.5);  // Speed of light  [meter/second]
double lam = c0/f0;  // Freespace Wavelength  [meter]
double t0  = 1/f0;  // Source Period  [second]

double dx = num_waves_x * lam / (Nx-1);
double dy = num_waves_y * lam / (Ny-1);
double dt = pow(pow(dx,-2) + pow(dy,-2), -0.5)/c0*.99;

double coef_eps_dx = dt/(eps0*dx);
double coef_eps_dy = dt/(eps0*dy);
double coef_mu_dx = dt/(mu0*dx);
double coef_mu_dy = dt/(mu0*dy);    

const int x_fi = 0;
const int x_li = Nx - 1;
const int y_fi = 0;
const int y_li = Ny - 1;

struct SimInfo
{
    int step;
    float f0;
    float dt;
};

using namespace std;
using value_t = float;

ofstream output_file;
bool do_logging = false;

#define ij_to_k(i, j, Nx) (Nx*(j) + (i)) 

__global__ void source_E(float *Ez, float dt, int step, int f0)
{
    // Source
    int source_i = Nx/3;
    int source_j = Ny/3;
    int source_k = ij_to_k(source_i, source_j, Nx);     
        
    //Ez[source_k] += sinf(2*M_PI*simInfo.f0*(simInfo.dt*simInfo.step)) * expf(-0.5*powf((simInfo.step-20)/8, 2));
    //Ez[source_k] += sin(2*M_PI*simInfo.f0*(simInfo.dt*simInfo.step)) * exp(-0.5*pow((simInfo.step-20)/8, 2));
    //Ez[source_k] += sin(2*M_PI*simInfo.step*simInfo.dt);
    //printf("========step: %d, k: %d============= \n", step, source_k);
    //printf("========step: %d, dt: %.9f, f0: %d============= \n", simInfo.step, simInfo.dt, temp_f0);
    Ez[source_k] += sin(2*M_PI*f0*(dt*step)) * exp(-0.5*pow((step-20)/8, 2));

}

__global__ void step_H(float *Hx, float *Hy, float *Ez,
    float coef_eps_dx, float coef_eps_dy, float coef_mu_dx, float coef_mu_dy)
{   
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int k_max = N - 1;
    if (k > k_max)
        return;

    int j = (int)(k / Nx);
    int i = (int)(k % Nx);
    int last_j = Ny - 1;
    int last_i = Nx - 1;            

    // Magnetic Field Update
    if (i != last_i && j != last_j)
    {
        Hx[k] -= coef_mu_dy * (Ez[k] - Ez[ ij_to_k(i, j+1, Nx) ]); 
        Hy[k] += coef_mu_dx * (Ez[k] - Ez[ ij_to_k(i+1, j, Nx) ]);
    }

    // for (int i=x_fi; i<x_li; i++)
    // {
    //     for (int j=y_fi; j<y_li; j++)
    //     {
    //         Hx[i][j] -= coef_mu_dy * (Ez[i][j] - Ez[i][j+1]); 
    //         Hy[i][j] += coef_mu_dx * (Ez[i][j] - Ez[i+1][j]);
    //     }
    // }
}

__global__ void step_E(float *Hx, float *Hy, float *Ez,
    float coef_eps_dx, float coef_eps_dy, float coef_mu_dx, float coef_mu_dy)
{
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int k_max = N - 1;
    if (k > k_max)
        return;

    int j = (int)(k / Nx);
    int i = (int)(k % Nx);
    int last_j = Ny - 1;
    int last_i = Nx - 1;            

    // Electric Field Update
    if (i != 0 && j != 0 && i != last_i && j != last_j)   
    {
        Ez[k] += coef_eps_dx*(Hy[ ij_to_k(i-1, j, Nx) ] - Hy[k]) - coef_eps_dy*(Hx[ ij_to_k(i, j-1, Nx) ] - Hx[k]);
    }   

    // for (int i=(x_fi+1); i<x_li; i++)
    // {
    //     for (int j=(y_fi+1); j<y_li; j++)
    //     {
    //         Ez[i][j] += coef_eps_dx*(Hy[i-1][j] - Hy[i][j]) - coef_eps_dy*(Hx[i][j-1] - Hx[i][j]);
    //     }
    // }    
}


int main()
{
    // logging
    struct passwd *pw = getpwuid(getuid());
    const char *c_homedir = pw->pw_dir;
    const string homedir = c_homedir;
    const string data_dir = homedir + "/.data";    
    cout << "data_dir: " << data_dir << "\n";
    const string output_file_path = data_dir +"/output_gpu.txt";
    cout << output_file_path << std::endl;
    output_file.open(output_file_path);    

    //gridDim()    
    int threads_per_block = 512;
    dim3 block_dim(threads_per_block, 1, 1);
    //int num_blocks = ceil(N / (float)block_dim.x);
    int num_blocks = (N+(threads_per_block-1)) / threads_per_block;    
    printf("num_blocks: %d \n", num_blocks);
    dim3 grid_dim(num_blocks, 1, 1);    

    printf("Nx: %d, Ny:%d, L0: %f, dx: %f, dt: %.9f \n", Nx, Ny, lam, dx, dt);

    float *Hx, *Hy, *Ez;
    float *d_Hx, *d_Hy, *d_Ez; 

    // Allocate host memory
    Hx = (float*)malloc(sizeof(float) * N);
    Hy = (float*)malloc(sizeof(float) * N);
    Ez = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        Hx[i] = 0.0f;
        Hy[i] = 0.0f;
        Ez[i] = 0.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_Hx, sizeof(float) * N);
    cudaMalloc((void**)&d_Hy, sizeof(float) * N);
    cudaMalloc((void**)&d_Ez, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_Hx, Hx, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Hy, Hy, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ez, Ez, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t start_logging, stop_logging;
    cudaEventCreate(&start_logging);
    cudaEventCreate(&stop_logging);

    SimInfo simInfo;
    simInfo.dt = dt;
    simInfo.f0 = f0;

    printf("simInfo.f0: %d \n", f0);

    float total_computation_time_ms = 0;
    int frame_capture_period = 5;
    float total_logging_time_ms = 0;
    output_file << Nx << "," << Ny << "," << Nt << "," << frame_capture_period << "\n";
    // Executing kernel 
    for (int i=0; i<Nt; i++)
    {
        simInfo.step = i;
        cudaEventRecord(start);
        source_E<<<1, 1>>>(d_Ez, dt, i, f0);
        step_H<<<grid_dim, block_dim>>>(d_Hx, d_Hy, d_Ez, coef_eps_dx, coef_eps_dy, coef_mu_dx, coef_mu_dy);
        step_E<<<grid_dim, block_dim>>>(d_Hx, d_Hy, d_Ez, coef_eps_dx, coef_eps_dy, coef_mu_dx, coef_mu_dy);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float step_computation_time_ms = 0;
        cudaEventElapsedTime(&step_computation_time_ms, start, stop);
        total_computation_time_ms += step_computation_time_ms;

        if (do_logging && (i % frame_capture_period == 0))
        {
            cudaEventRecord(start_logging);
            // Transfer data back to host memory
            //cudaMemcpy(Hx, d_Hx, sizeof(float) * N, cudaMemcpyDeviceToHost);
            //cudaMemcpy(Hy, d_Hy, sizeof(float) * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(Ez, d_Ez, sizeof(float) * N, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop_logging);
            cudaEventSynchronize(stop_logging);
            float logging_time_ms = 0;
            cudaEventElapsedTime(&logging_time_ms, start_logging, stop_logging);
            total_logging_time_ms += logging_time_ms;
            
            // copy frames to the output file
            for (int k=0; k<N; k++)
            {
                value_t value_Ez = Ez[k];
                output_file << value_Ez;
                if (k % N == (N-1)) // a frame ended            
                {
                    output_file << ";";
                }
                else
                {
                    output_file << ",";
                }            
            }

        }
    }
    //vector_add<<<grid_dim, block_dim>>>(d_out, d_a, d_b, N);
    //vector_add_plain<<<1, 1>>>(d_out, d_a, d_b, N);

    
    printf("Total Computation Time %f ms \n", total_computation_time_ms);    
    printf("Total Logging Time %f ms \n", total_logging_time_ms);   
    
    //std::cout << "test" << "\n";
    //std::cout <<out[0] << "\n";

    // Verification
    // for(int i = 0; i < N; i++){
    //     bool is_ok = fabs(out[i] - a[i] - b[i]) < MAX_ERR;
    //     if (!is_ok)
    //     {
    //         printf("error!, i: %d \n", i);
    //         break;
    //     }
    //     //assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    // }
    
    // printf("out[0] = %f\n", out[0]);
    // std::cout << "passed" << "\n";    

    // Deallocate device memory
    cudaFree(Hx);
    cudaFree(Hy);
    cudaFree(Ez);

    // Deallocate host memory
    free(Hx); 
    free(Hy); 
    free(Ez);

    return 0;
}