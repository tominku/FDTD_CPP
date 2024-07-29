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
    std::string name;
    int total_steps;  
    int num_threads;     
    std::string material_file_path;
    std::vector<json> probes;

private:        
    void init()
    {                
        // std::string config_file_name = "config.json";
        // FileManager &fileManager = FileManager::instance();
        // auto current_dir_path = fileManager.get_current_dir_path();
        // std::string file_path = fileManager.convert_to_path(current_dir_path, config_file_name);        
        
        // json j;
        // std::ifstream is(file_path);        
        // is >> j;   

        // name = j["name"];
        // total_steps = j["total_steps"];
        // num_threads = j["num_threads"];
        // material_file_path = j["material_file_path"];
        // probes = j["probes"].template get<std::vector<json>>();

    }
    Config() { init(); }

protected:
    std::string toName()
    {
        return "Config";
    }

public:            
   void load(std::string config_file_name)
   {        
        FileManager &fileManager = FileManager::instance();
        auto current_dir_path = fileManager.get_current_dir_path();
        std::string file_path = fileManager.convert_to_path(current_dir_path, config_file_name);        
        
        json j;
        std::ifstream is(file_path);        
        is >> j;   

        name = j["name"];
        total_steps = j["total_steps"];
        num_threads = j["num_threads"];
        material_file_path = j["material_file_path"];
        probes = j["probes"].template get<std::vector<json>>();    
        
        fileManager.create_output_dir(name);
   }

   void save_sim_frames(json j, std::string file_name)
   {
        FileManager &fileManager = FileManager::instance();
        //fileManager.
        auto output_dir_path = fileManager.get_output_dir_path();        
        auto output_file_path = output_dir_path / file_name;
        fileManager.save_json(j, output_file_path);   
   }

   static Config& instance()
   {
      static Config INSTANCE;
      return INSTANCE;
   }    
};