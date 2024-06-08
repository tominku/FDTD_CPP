// OpenMP header
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 10
#define COLS 15 
#define R_FI 0 // row first index
#define R_LI (ROWS-1)  // row last index
#define C_FI 0 // col first index
#define C_LI (COLS-1)  // col last index

int main()
{
    printf("R_LI %d \n", R_LI);
    double Ez[ROWS][COLS] = {0, };
    double Hx[ROWS][COLS] = {0, };
    double Hy[ROWS][COLS] = {0, };

    for (int i=R_FI; i<R_LI; i++)
    {
        for (int j=C_FI; j<C_LI; j++)
        {
            int udx = 1;
            Hx[i][j] += udx * (Ez[i][j] - Ez[i][j+1]); 
            Hy[][]
        }
    }

    /*
    int arr[4] = {9, };
    omp_set_num_threads(4);
	int nthreads, tid;
	// Begin of parallel region
	#pragma omp parallel private(tid, nthreads)
	{
		// Getting thread number
		tid = omp_get_thread_num();
		printf("OMP thread = %d\n",
			tid);
        arr[tid] = tid;
		if (tid == 0) {
			// Only master thread does this
			nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n",
				nthreads);
		}
	}
    for (int i=0; i<4; i++)
    {
        printf("element at %d: %d\n", i, arr[i]);
    }
    */
}
