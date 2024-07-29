#pragma once
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
#include "Base.h"
#include "FileManager.h"

struct MaterialData
{
    bool has_material;
    float *origin_data;
    int origin_height;
    int origin_width;
    int origin_num_pixels;
    float *scaled_data;
};

using namespace std;

class Material : Base
{

private:
    string file_path;
    MaterialData material_data;
    int num_pixels;
    bool has_material;

protected:
    std::string toName()
    {
        return "Material";
    }

public:

    // Material(string file_)
    // {
    //     has_material = false;
    //     FileManager &fileManager = FileManager::instance();
    //     auto current_dir_path = fileManager.get_current_dir_path();
    //     file_path = fileManager.convert_to_path(current_dir_path, file_);        
    //     material_data.origin_data = NULL;
    // }

    Material()
    {
        has_material = false;
        Config &config = Config::instance();
        file_path = config.material_file_path;   
        material_data.origin_data = NULL;
        int N = Nx * Ny;

        MaterialData material_data = parse();       

        json j;
        j["has_material"] = material_data.has_material;
        if (material_data.has_material)
        {    
            std::vector<int> material_values(material_data.scaled_data, material_data.scaled_data+N);
            int vec_size = material_values.size();
            assert (vec_size == N);        
            j["material_data_size"] = vec_size;
            j["material_data"] = material_values;
        }
        j["Nx"] = Nx;
        j["Ny"] = Ny;
        
        config.save_material(j, "material.json");
        // std::string path = fileManager.into_data_dir("material.json");
        // fileManager.save_json(j, path);                    
    }

   static Material& instance()
   {
      static Material INSTANCE;
      return INSTANCE;
   }    

    bool hasMaterial()
    {
        return has_material;
    }

    MaterialData getMaterialData()
    {
        return material_data;
    }

    MaterialData scaleToFit(int nx, int ny)
    {
        int N = nx * ny;
        float *scaled_data = new float[N];

        #pragma omp parallel for num_threads(6) collapse(2) if(true)   
        for (int i=0; i<nx; i++)
        {        
            for (int j=0; j<ny; j++)
            {
                float float_i = i / (float)(nx - 1);
                float float_j = j / (float)(ny - 1);
                int origin_i = (int)round((material_data.origin_height - 1)*float_i);
                int origin_j = (int)round((material_data.origin_width - 1)*float_j);
                int origin_height = material_data.origin_height;
                float pixel_value = material_data.origin_data[ij_to_k_(origin_i, origin_j, origin_height)];
                int k_for_ij = ij_to_k(i, j);
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
        bool is_file_ok = fileManager.get_json(file_path, material_json);
        if (!is_file_ok)
        {
            material_data.has_material = false;
            return material_data;
        }
        
        material_data.origin_width = material_json["width"];
        material_data.origin_height = material_json["height"];
        material_data.origin_num_pixels = material_data.origin_width * material_data.origin_height;        

        std::vector<float> data_vector = material_json["data"].template get<std::vector<float>>();         
        int data_size = data_vector.size();                     
        assert (data_size == material_data.origin_num_pixels);        
        material_data.origin_data = new float[data_size];        

        std::copy(data_vector.begin(), data_vector.end(), material_data.origin_data);

        material_data.has_material = true;
        MaterialData material_data_ = scaleToFit(Nx, Ny);        
        return material_data_;   
    }
};