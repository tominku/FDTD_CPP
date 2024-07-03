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
#include <filesystem>
#include <cassert>
#include <cstdlib>
#include "macros.h"
#include "Base.h"
#include "FileManager.h"

struct MaterialData
{
    int *origin_data;
    int origin_height;
    int origin_width;
    int origin_num_pixels;
    int *scaled_data;
};

using namespace std;

class Material : Base
{

private:
    string file_path;
    MaterialData material_data;
    int num_pixels;

protected:
    std::string toName()
    {
        return "Material";
    }

public:
    Material(string file_)
    {
        FileManager &fileManager = FileManager::instance();
        auto current_dir_path = fileManager.get_current_dir_path();
        file_path = fileManager.convert_to_path(current_dir_path, file_);        
        material_data.origin_data = NULL;
    }

    MaterialData scaleToFit(int Nx, int Ny)
    {
        int N = Nx * Ny;
        int *scaled_data = new int[N];

        #pragma omp parallel for num_threads(6) collapse(2) if(true)   
        for (int i=0; i<Nx; i++)
        {        
            for (int j=0; j<Ny; j++)
            {
                float float_i = i / (float)(Nx - 1);
                float float_j = j / (float)(Ny - 1);
                int origin_i = (int)round((material_data.origin_height - 1)*float_i);
                int origin_j = (int)round((material_data.origin_width - 1)*float_j);
                int origin_height = material_data.origin_height;
                int pixel_value = material_data.origin_data[ij_to_k(origin_i, origin_j, origin_height)];
                int k_for_ij = ij_to_k(i, j, Nx);
                scaled_data[k_for_ij] = pixel_value;
            }
        }

        // for (int k=0; k<N; ++k)
        // {
        //     cout << scaled_data[k];
        // }
        material_data.scaled_data = scaled_data;
        return material_data;
    }

    MaterialData parse()
    {        
        FileManager &fileManager = FileManager::instance(); 
        json material_json;               
        fileManager.get_json(file_path, material_json);
        
        material_data.origin_width = material_json["width"];
        material_data.origin_height = material_json["height"];
        material_data.origin_num_pixels = material_data.origin_width * material_data.origin_height;        

        std::vector<int> data_vector = material_json["data"].template get<std::vector<int>>();         
        int data_size = data_vector.size();                     
        assert (data_size == material_data.origin_num_pixels);        
        material_data.origin_data = new int[data_size];        

        std::copy(data_vector.begin(), data_vector.end(), material_data.origin_data);

        return material_data;
    }
};