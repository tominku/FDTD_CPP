#pragma once

// #if !defined( USE_GPU )
// #define USE_GPU 0
// #endif

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#define NUM_THREADS 6

using value_t = float;
using namespace std;
using namespace std::chrono;

#define ij_to_k(i, j, Nx) (Nx*(j) + (i))

#define devisions_per_wave 10  // Divisions per Wavelength   [unitless]
#define num_waves_x 15 //  # wave lengths in x-dir [unitless]
#define num_waves_y 30 //  # wave lengths in y-dir 
#define Nx (num_waves_x*devisions_per_wave + 1)
#define Ny (num_waves_y*devisions_per_wave + 1)

const int x_fi = 0;
const int x_li = Nx - 1;
const int y_fi = 0;
const int y_li = Ny - 1;

const int n_PML_X = 10;
const int n_PML_Y = 10;

// Define Simulation Based off Source and Wavelength
int f0 = 1e6; // Frequency of Source  [Hertz]
int nt = 2000; // Number of time steps  [unitless]

// Spatial and Temporal System
value_t eps0 = 8.854 * 1e-12;  // Permittivity of vacuum [farad/meter]
value_t mu0 = 4*M_PI* 1e-7;  // Permeability of vacuum [henry/meter]
value_t c0 = 1/pow((eps0*mu0), 0.5);  // Speed of light  [meter/second]
value_t lam = c0/f0;  // Freespace Wavelength  [meter]
value_t t0  = 1/f0;  // Source Period  [second]

value_t space_size_x = num_waves_x * lam;
value_t space_size_y = num_waves_y * lam;
value_t dx = space_size_x / (Nx-1);
value_t dy = space_size_y / (Ny-1);
value_t dt = pow(pow(dx,-2) + pow(dy,-2), -0.5)/c0*.99;

value_t coef_eps_dx = dt/(eps0*dx);
value_t coef_eps_dy = dt/(eps0*dy);
value_t coef_mu_dx = dt/(mu0*dx);
value_t coef_mu_dy = dt/(mu0*dy);

void initialize_zero(value_t *values, int len)
{
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int i=0; i<len; ++i)
    {
        values[i] = 0;
    }
}

/*
[Nx,Ny] = deal(Lx*Lf,Ly*Lf);    % Points in x,y           [unitless]
x  = linspace(0,Lx,Nx+1)*L0;    % x vector                [meter]
y  = linspace(0,Ly,Ny+1)*L0;    % y vector                [meter]
[dx,dy] = deal(x(2),y(2));      % x,y,z increment         [meter]
dt = (dx^-2+dy^-2)^-.5/c0*.99;  % Time step CFL condition [second]
*/