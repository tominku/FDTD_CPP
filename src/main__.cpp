// OpenMP header
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <chrono> 
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

using namespace std;
using namespace std::chrono;

#define NUM_THREADS 2
#define PRINT 0

#define devisions_per_wave 10  // Divisions per Wavelength   [unitless]
#define num_waves_x 8 //  # wave lengths in x-dir [unitless]
#define num_waves_y 8 //  # wave lengths in y-dir 
#define Nx (num_waves_x*devisions_per_wave + 1)
#define Ny (num_waves_y*devisions_per_wave + 1)

const int x_fi = 0;
const int x_li = Nx - 1;
const int y_fi = 0;
const int y_li = Ny - 1;

ofstream output_file;

void step_em_fields(double Hx[][Ny], double Hy[][Ny], double Ez[][Ny],
    double coef_eps_dx, double coef_eps_dy, double coef_mu_dx, double coef_mu_dy)
{   
    // Magnetic Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (int i=x_fi; i<x_li; i++)
    {
        for (int j=y_fi; j<y_li; j++)
        {
            int udx = 1;
            Hx[i][j] -= coef_mu_dy * (Ez[i][j] - Ez[i][j+1]); 
            Hy[i][j] += coef_mu_dx * (Ez[i][j] - Ez[i+1][j]);
            if (PRINT)
                printf("M-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
    // Electric Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2)
    for (int i=(x_fi+1); i<x_li; i++)
    {
        for (int j=(y_fi+1); j<y_li; j++)
        {
            Ez[i][j] += coef_eps_dx*(Hy[i-1][j] - Hy[i][j]) - coef_eps_dy*(Hx[i][j-1] - Hx[i][j]);
            if (PRINT)
                printf("E-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
}

int main__()
{
    struct passwd *pw = getpwuid(getuid());
    const char *c_homedir = pw->pw_dir;
    const string homedir = c_homedir;
    const string data_dir = homedir + "/.data";
    cout << "data_dir: " << data_dir << "\n";
    const string output_file_path = data_dir +"/output.txt";
    output_file.open(output_file_path);
    // Define Simulation Based off Source and Wavelength
    int f0 = 1e6; // Frequency of Source  [Hertz]
    int nt = 1000; // Number of time steps  [unitless]

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
    /*
    [Nx,Ny] = deal(Lx*Lf,Ly*Lf);    % Points in x,y           [unitless]
    x  = linspace(0,Lx,Nx+1)*L0;    % x vector                [meter]
    y  = linspace(0,Ly,Ny+1)*L0;    % y vector                [meter]
    [dx,dy] = deal(x(2),y(2));      % x,y,z increment         [meter]
    dt = (dx^-2+dy^-2)^-.5/c0*.99;  % Time step CFL condition [second]
    */

    //printf("R_LI %d \n", x_li);
    printf("Nx: %d, Ny:%d, L0: %f, dx: %f, dt: %.9f \n", Nx, Ny, lam, dx, dt);

    int computation_time = 0; 

    double Ez[Nx][Ny] = {0, };
    double Hx[Nx][Ny] = {0, };
    double Hy[Nx][Ny] = {0, };

    output_file << Nx << "," << Ny << "," << nt << "\n";
    for (int step=0; step<nt; step++)
    {        
        //Point Source
        //Ez[round(ROWS/2),round(COLS/2)] += sin(2*pi*f0*dt*t).*exp(-.5*((step-20)/8)^2);
        Ez[(int)round(Nx/3)][(int)round(Ny/3)] += sin(2*M_PI*f0*(dt*step)) * exp(-0.5*pow((step-20)/8, 2));
        
        auto t1 = steady_clock::now();
        
        step_em_fields(Hx, Hy, Ez, coef_eps_dx, coef_eps_dy, coef_mu_dx, coef_mu_dy);        
        
        auto t2 = steady_clock::now();

        auto duration = duration_cast<microseconds>(t2 - t1);
        computation_time += duration.count();

        for (int i=x_fi; i<=x_li; i++)
        {
            for (int j=y_fi; j<=y_li; j++)
            {
                output_file << Ez[i][j]; 
                if (j < y_li)
                    output_file << ",";
            }
            if (i < x_li)
                output_file << "\n";
        }
        if (step < (nt - 1))
            output_file << "\n*\n";
    }
    
    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << "computation_time: " << computation_time / 1000 << " ms" << std::endl;

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
