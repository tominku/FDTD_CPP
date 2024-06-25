#include <stdio.h>

__global__
void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

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

int main()
{
    helloCUDA<<<1, 1>>>();
    Test test;
    test.Print();
    cudaDeviceSynchronize();        
    return 0;
}