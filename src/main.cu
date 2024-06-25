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

struct StructTest
{
    int a = 0;
    int b = 1;
};

int main()
{
    helloCUDA<<<1, 1>>>();
    Test test;
    test.Print();

    StructTest st;
    printf("struct test: %d \n", st.b);

    cudaDeviceSynchronize();        

    return 0;
}