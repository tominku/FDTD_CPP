#pragma once

// #if !defined( USE_GPU )
// #define USE_GPU 0
// #endif

#include "nlohmann/json.hpp"
using json = nlohmann::json;

using value_t = float;

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
