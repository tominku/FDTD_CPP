// OpenMP header
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <chrono>
using namespace std::chrono;

#define ROWS 500
#define COLS 500
#define R_FI 0 // row first index
#define R_LI (ROWS-1)  // row last index
#define C_FI 0 // col first index
#define C_LI (COLS-1)  // col last index
#define NUM_THREADS 2
#define PRINT 0

double Ez[ROWS][COLS] = {0, };
double Hx[ROWS][COLS] = {0, };
double Hy[ROWS][COLS] = {0, };

void step_em_fields()
{
    // Magnetic Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (int i=R_FI; i<R_LI; i++)
    {
        for (int j=C_FI; j<C_LI; j++)
        {
            int udx = 1;
            Hx[i][j] -= udx * (Ez[i][j] - Ez[i][j+1]); 
            Hy[i][j] += udx * (Ez[i][j] - Ez[i+1][j]);
            if (PRINT)
                printf("M-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
    // Electric Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (int i=(R_FI+1); i<R_LI; i++)
    {
        for (int j=(C_FI+1); j<C_LI; j++)
        {
            Ez[i][j] += (Hy[i-1][j] - Hy[i][j]) + (Hx[i][j-1] - Hx[i][j]);
            if (PRINT)
                printf("E-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
}

int main()
{
    // Define Simulation Based off Source and Wavelength
    int f0 = 1e6; // Frequency of Source  [Hertz]
    int Lf = 10;  // Divisions per Wavelength   [unitless]
    double Lx = 8; //  # wave lengths in x-dir [unitless]
    double Ly = 8; //  # wave lengths in y-dir 
    int nt = 1000; // Number of time steps  [unitless]

    // Spatial and Temporal System
    double e0 = 8.854 * 1e-12;  // Permittivity of vacuum [farad/meter]
    double u0 = 4*M_PI* 1e-7;  // Permeability of vacuum [henry/meter]
    double c0 = 1/pow((e0*u0), 0.5);  // Speed of light  [meter/second]
    double L0 = c0/f0;  // Freespace Wavelength  [meter]
    double t0  = 1/f0;  // Source Period  [second]

    int Nx = Lx*Lf +1;
    int Ny = Ly*Lf + 1;
    double dx = Lx * L0 / (Nx-1);
    double dy = Ly * L0 / (Ny-1);
    double dt = pow(pow(dx,-2) + pow(dy,-2), -0.5)/c0*.99;
    /*
    [Nx,Ny] = deal(Lx*Lf,Ly*Lf);    % Points in x,y           [unitless]
    x  = linspace(0,Lx,Nx+1)*L0;    % x vector                [meter]
    y  = linspace(0,Ly,Ny+1)*L0;    % y vector                [meter]
    [dx,dy] = deal(x(2),y(2));      % x,y,z increment         [meter]
    dt = (dx^-2+dy^-2)^-.5/c0*.99;  % Time step CFL condition [second]
    */

    printf("R_LI %d \n", R_LI);
    printf("L0: %f, dx: %f, dt: %.9f \n", L0, dx, dt);

    auto start = high_resolution_clock::now();

    for (int step=0; step<100; step++)
    {
        //Point Source
        //Ez[round(ROWS/2),round(COLS/2)] += sin(2*pi*f0*dt*t).*exp(-.5*((step-20)/8)^2);
        Ez[(int)round(ROWS/2)][(int)round(COLS/2)] += sin(2*M_PI*f0*(dt*step)) * exp(-0.5*pow((step-20)/8, 2));
        
        step_em_fields();
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    
    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << "duration: " << duration.count() << " ms" << std::endl;

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
