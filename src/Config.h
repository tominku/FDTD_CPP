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
#include "nlohmann/json.hpp"
using json = nlohmann::json;


class Config : Base
{
public:
    int total_steps;    
private:        
    void init()
    {                
        std::string config_file_name = "config.json";
        FileManager &fileManager = FileManager::instance();
        auto current_dir_path = fileManager.get_current_dir_path();
        std::string file_path = fileManager.convert_to_path(current_dir_path, config_file_name);        
        
        json j;
        std::ifstream is(file_path);        
        is >> j;   

        total_steps = j["total_steps"];
        std::vector<json> probes = j["probes"].template get<std::vector<json>>();

    }
    Config() { init(); }

protected:
    std::string toName()
    {
        return "Config";
    }

public:            
   
   static Config& instance()
   {
      static Config INSTANCE;
      return INSTANCE;
   }    
};