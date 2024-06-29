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

struct MaterialData
{
    int *data;
    int height;
    int width;
    int num_pixels;
};

using namespace std;

class Material
{

private:
    string file;
    MaterialData material_data;
    int num_pixels;

public:
    Material(string file_)
    {
        file = file_;
        material_data.data = NULL;
    }

    MaterialData parse()
    {
        struct passwd *pw = getpwuid(getuid());
        const char *c_homedir = pw->pw_dir;
        const string homedir = c_homedir;
        const string data_dir = homedir + "/.data/";
        
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
        material_data.height = height;
        material_data.width = width;
        material_data.num_pixels = num_pixels;
        material_data.data =  new int[num_pixels];
        cout << "material height: " << height << ", material width: " << width <<endl;
        //assert (num_pixels == data_string);
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
                material_data.data[char_count] = current_data;
                //cout << material_data.data[char_count];
                char_count += 1;
            }
            pos += 1;
        }

        for (int i=0; i<num_pixels; ++i)
        {
            //cout << material_data[i];
        }

        return material_data;
    }
};