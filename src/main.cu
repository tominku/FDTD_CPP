#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 1000000
#define MAX_ERR 1e-6

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

int main()
{
    //gridDim()    
    dim3 block_dim(512, 1, 1);
    int num_blocks = ceil(N / (float)block_dim.x);
    printf("a: %f \n", N / (float)block_dim.x);
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

    // Executing kernel 
    vector_add<<<grid_dim, block_dim>>>(d_out, d_a, d_b, N);
    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    printf("Test \n");    
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