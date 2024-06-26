#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_ERR 1e-6

#define devisions_per_wave 10  // Divisions per Wavelength   [unitless]
#define num_waves_x 48 //  # wave lengths in x-dir [unitless]
#define num_waves_y 48 //  # wave lengths in y-dir 
#define Nx (num_waves_x*devisions_per_wave + 1)
#define Ny (num_waves_y*devisions_per_wave + 1)
#define N (Nx*Ny)
#define Nt 1000

const int x_fi = 0;
const int x_li = Nx - 1;
const int y_fi = 0;
const int y_li = Ny - 1;

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

__global__ void vector_add_plain(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int last_index = n - 1;
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i > last_index)
        return;

    out[i] = a[i] + b[i];    
}

#define ij_to_k(i, j, Nx) ((j)*Nx + (i)) 

//__global__ void step_EM(int Nx, int Ny, int N, double *Hx, double *Hy, double *Ez,
__global__ void step_EM(float *Hx, float *Hy, float *Ez,
    float coef_eps_dx, float coef_eps_dy, float coef_mu_dx, float coef_mu_dy)
{   
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    int k_max = N - 1;
    if (k > k_max)
        return;

    int j = k / Nx;
    int i = k % Nx;
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
    //gridDim()    
    int threads_per_block = 512;
    dim3 block_dim(threads_per_block, 1, 1);
    //int num_blocks = ceil(N / (float)block_dim.x);
    int num_blocks = (N+(threads_per_block-1)) / threads_per_block;    
    printf("num_blocks: %d \n", num_blocks);
    dim3 grid_dim(num_blocks, 1, 1);    

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

    cudaEventRecord(start);
    // Executing kernel 
    for (int i=0; i<Nt; i++)
    {
        step_EM<<<grid_dim, block_dim>>>(d_Hx, d_Hy, d_Ez, coef_eps_dx, coef_eps_dy, coef_mu_dx, coef_mu_dy);
    }
    //vector_add<<<grid_dim, block_dim>>>(d_out, d_a, d_b, N);
    //vector_add_plain<<<1, 1>>>(d_out, d_a, d_b, N);
    cudaEventRecord(stop);

    // Transfer data back to host memory
    cudaMemcpy(Hx, d_Hx, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(Hy, d_Hy, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ez, d_Ez, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed Time %f ms \n", milliseconds);    
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