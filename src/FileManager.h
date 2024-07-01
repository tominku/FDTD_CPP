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

class FileManager : Base
{
private:
    std::ofstream output_file;
    std::ofstream output_material_file;
    std::string output_file_name;
    std::string output_material_file_name;
    fs::path data_dir_path;
    fs::path material_file_path;    
    
    FileManager() { }

protected:
    std::string toName()
    {
        return "FileUtil";
    }

public:

    void init(std::string output_file_name_, std::string output_material_file_)
    {
        output_file_name = output_file_name_;
        output_material_file_name = output_material_file_;
        const std::string str_home_path = getenv("HOME");     
        auto home_dir_path = fs::path(str_home_path);                
        data_dir_path = home_dir_path / ".data";
        fs::create_directory(data_dir_path);
        assert(!fs::create_directory(data_dir_path));    
        
        std::string msg = fmt::format("data_dir_path: {}", data_dir_path.c_str());
        print(msg);
        
        auto output_file_path = data_dir_path / output_file_name;
        msg = fmt::format("output_file_path: {}", output_file_path.c_str());
        print(msg);

        auto material_file_path = data_dir_path / output_material_file_name;
        msg = fmt::format("material_file_path: {}", material_file_path.c_str());
        print(msg);
    }
   
   static FileManager& instance()
   {
      static FileManager INSTANCE;
      return INSTANCE;
   }    
};