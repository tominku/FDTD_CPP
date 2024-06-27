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

#define NUM_THREADS 10
#define PRINT 0

#define devisions_per_wave 10  // Divisions per Wavelength   [unitless]
#define num_waves_x 48 //  # wave lengths in x-dir [unitless]
#define num_waves_y 48 //  # wave lengths in y-dir 
#define Nx (num_waves_x*devisions_per_wave + 1)
#define Ny (num_waves_y*devisions_per_wave + 1)

typedef float value_t;

const int x_fi = 0;
const int x_li = Nx - 1;
const int y_fi = 0;
const int y_li = Ny - 1;

const int n_PML_X = 10;
const int n_PML_Y = 10;

ofstream output_file;

bool do_parallel = true;

#define ij_to_k(i, j, Nx) (Nx*(j) + (i))

void step_em_pml(value_t *Hx, value_t *Hy, value_t *Ez,
    value_t coef_eps_dx, value_t coef_eps_dy, value_t coef_mu_dx, value_t coef_mu_dy)
{   

    // Magnetic Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2) if(do_parallel)   
    for (int i=x_fi; i<x_li; i++)
    {        
        for (int j=y_fi; j<y_li; j++)
        {
            
            int k_for_ij = ij_to_k(i, j, Nx);
            int k_for_ijp1 = ij_to_k(i, j+1, Nx);
            int k_for_ip1j = ij_to_k(i+1, j, Nx); 
            
            Hx[k_for_ij] -= coef_mu_dy * (Ez[k_for_ij] - Ez[k_for_ijp1]); 
            Hy[k_for_ij] += coef_mu_dx * (Ez[k_for_ij] - Ez[k_for_ip1j]);
            // Hx[i][j] -= coef_mu_dy * (Ez[i][j] - Ez[i][j+1]); 
            // Hy[i][j] += coef_mu_dx * (Ez[i][j] - Ez[i+1][j]);
            if (PRINT)
                printf("M-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
    // Electric Field Update
    #pragma omp parallel for num_threads(NUM_THREADS) collapse(2) if(do_parallel)
    for (int i=(x_fi+1); i<x_li; i++)
    {
        for (int j=(y_fi+1); j<y_li; j++)
        {
            int k_for_ij = ij_to_k(i, j, Nx);
            int k_for_ijm1 = ij_to_k(i, j-1, Nx);
            int k_for_im1j = ij_to_k(i-1, j, Nx); 

            Ez[k_for_ij] += coef_eps_dx*(Hy[k_for_im1j] - Hy[k_for_ij]) - coef_eps_dy*(Hx[k_for_ijm1] - Hx[k_for_ij]);
            if (PRINT)
                printf("E-Field i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
        }
    }
}

int main()
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
    value_t eps0 = 8.854 * 1e-12;  // Permittivity of vacuum [farad/meter]
    value_t mu0 = 4*M_PI* 1e-7;  // Permeability of vacuum [henry/meter]
    value_t c0 = 1/pow((eps0*mu0), 0.5);  // Speed of light  [meter/second]
    value_t lam = c0/f0;  // Freespace Wavelength  [meter]
    value_t t0  = 1/f0;  // Source Period  [second]

    value_t dx = num_waves_x * lam / (Nx-1);
    value_t dy = num_waves_y * lam / (Ny-1);
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

    //printf("R_LI %d \n", x_li);
    printf("Nx: %d, Ny:%d, L0: %f, dx: %f, dt: %.9f \n", Nx, Ny, lam, dx, dt);

    int computation_time = 0; 
    int N = Nx * Ny;
    value_t *Ez = new value_t[N];
    value_t *Hx = new value_t[N];
    value_t *Hy = new value_t[N];

    output_file << Nx << "," << Ny << "," << nt << "\n";
    for (int step=0; step<nt; step++)
    {        
        //Point Source
        //Ez[round(ROWS/2),round(COLS/2)] += sin(2*pi*f0*dt*t).*exp(-.5*((step-20)/8)^2);
        int source_k = ij_to_k(Nx/3, Ny/3, Nx);
        Ez[source_k] += sin(2*M_PI*f0*(dt*step)) * exp(-0.5*pow((step-20)/8, 2));
        
        auto t1 = steady_clock::now();
        
        step_em_pml(Hx, Hy, Ez, coef_eps_dx, coef_eps_dy, coef_mu_dx, coef_mu_dy);        
        
        auto t2 = steady_clock::now();

        auto duration = duration_cast<microseconds>(t2 - t1);
        computation_time += duration.count();

        for (int i=x_fi; i<=x_li; i++)
        {
            for (int j=y_fi; j<=y_li; j++)
            {
                int k_for_ij = ij_to_k(i, j, Nx);
                output_file << Ez[k_for_ij]; 
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
