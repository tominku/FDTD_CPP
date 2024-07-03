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
#include "nlohmann/json.hpp"
using json = nlohmann::json;

class FileManager : Base
{
private:    
    fs::path data_dir_path;    
    
    FileManager() { }

protected:
    std::string toName()
    {
        return "FileUtil";
    }

public:
    
    void init()
    {                
        const std::string str_home_path = getenv("HOME");     
        auto home_dir_path = fs::path(str_home_path);                
        data_dir_path = home_dir_path / ".data";
        fs::create_directory(data_dir_path);
        assert(!fs::create_directory(data_dir_path));    
        
        std::string msg = fmt::format("data_dir_path: {}", data_dir_path.c_str());
        print(msg);                
    }

    void get_json(std::string file_path, json &j)
    {
        std::ifstream is(file_path);        
        is >> j;
    }

    std::string get_data_dir_path()
    {
        return data_dir_path.c_str();
    }        

    fs::path get_current_dir_path()
    {
        auto cur_path = fs::current_path();
        return cur_path.c_str();
    }

    std::string convert_to_path(std::string file_name)
    {
        auto path = data_dir_path / file_name;
        return path.c_str();
    }

    std::string convert_to_path(fs::path dir_path, std::string file_name)
    {
        auto path = dir_path / file_name;
        return path.c_str();
    }

    void save_json(json &j, std::string path)
    {
        std::ofstream o(path);
        o << j;
    }
   
   static FileManager& instance()
   {
      static FileManager INSTANCE;
      return INSTANCE;
   }    
};