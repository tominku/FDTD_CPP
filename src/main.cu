#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 10000000
#define MAX_ERR 1e-6

#define devisions_per_wave 10  // Divisions per Wavelength   [unitless]
#define num_waves_x 80 //  # wave lengths in x-dir [unitless]
#define num_waves_y 80 //  # wave lengths in y-dir 
#define Nx (num_waves_x*devisions_per_wave + 1)
#define Ny (num_waves_y*devisions_per_wave + 1)

const int x_fi = 0;
const int x_li = Nx - 1;
const int y_fi = 0;
const int y_li = Ny - 1;

class Test
{
private:
    int a;
public:
    Test()
    {
        a = 3;
    }
    void Print()
    {
        printf("class test %d \n", a);
    }

};

struct StructTest
{
    int a = 0;
    int b = 1;
};


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

__global__ void step_EM(int Nx, int Ny, int N, double *Hx, double *Hy, double *Ez,
    double coef_eps_dx, double coef_eps_dy, double coef_mu_dx, double coef_mu_dy)
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

    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Executing kernel 
    vector_add<<<grid_dim, block_dim>>>(d_out, d_a, d_b, N);
    //vector_add_plain<<<1, 1>>>(d_out, d_a, d_b, N);
    cudaEventRecord(stop);

    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed Time %f ms \n", milliseconds);    
    //std::cout << "test" << "\n";
    std::cout <<out[0] << "\n";

    // Verification
    for(int i = 0; i < N; i++){
        bool is_ok = fabs(out[i] - a[i] - b[i]) < MAX_ERR;
        if (!is_ok)
        {
            printf("error!, i: %d \n", i);
            break;
        }
        //assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    
    printf("out[0] = %f\n", out[0]);
    std::cout << "passed" << "\n";    

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);

    return 0;
}