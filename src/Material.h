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
    string file;
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
        file = file_;
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
        struct passwd *pw = getpwuid(getuid());
        const char *c_homedir = pw->pw_dir;
        const string homedir = c_homedir;
        //const string data_dir = homedir + "/.data/";        
        const string data_dir = "data/";
      
        cout << "material data_dir: " << data_dir << "\n";
        const string file_path = data_dir + file;
        std::cout << "material file path: " << file_path << std::endl;
        
        ifstream material_file(file_path.data());
        string data_string;
        getline(material_file, data_string);
        size_t pos = data_string.find(';');
        string str_data_info = data_string.substr(0, pos-1+1);
        cout << "pos: " << pos << endl;
        cout << "data info: " << str_data_info << endl;
        cout << "data info str len: " << str_data_info.size() << endl;
        pos = str_data_info.find(',');
        string str_height = str_data_info.substr(0, pos-1+1);
        string str_width = str_data_info.substr(pos + 1, str_data_info.size() - str_height.size() - 1);
        int height = stoi(str_height);
        int width = stoi(str_width);
        int num_pixels = height * width;
        material_data.origin_height = height;
        material_data.origin_width = width;
        material_data.origin_num_pixels = num_pixels;
        material_data.origin_data =  new int[num_pixels];
        cout << "material height: " << height << ", material width: " << width <<endl;

        pos = str_data_info.size() + 1;
        int char_count = 0;
        while(true)
        {
            if (pos > (data_string.size() - 1))
                break;
            char current_char = data_string[pos];
            if (current_char != ',') 
            {   
                int current_data = current_char - '0';
                material_data.origin_data[char_count] = current_data;
                //cout << material_data.data[char_count];
                char_count += 1;
            }
            pos += 1;
        }
        
        assert (num_pixels == char_count);
        
        // for (int i=0; i<num_pixels; ++i)
        // {
        //     //cout << material_data[i];
        // }

        return material_data;
    }
};