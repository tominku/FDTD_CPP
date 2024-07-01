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

namespace fs = std::filesystem;

class FileUtil
{
private:
    fs::path data_dir_path;    

    void init()
    {
        const std::string str_home_path = getenv("HOME");     
        auto home_dir_path = fs::path(str_home_path);                
        data_dir_path = home_dir_path / ".data";
        fs::create_directory(data_dir_path);
        assert(!fs::create_directory(data_dir_path));    
        
        std::cout << "[FileUtil] data_dir: " << data_dir_path << std::endl;
        auto output_file_path = data_dir_path / "output_cpu.txt";
        std::cout << "[FileUtil] output_path: " <<output_file_path << std::endl;
    }

public:
    FileUtil()
    {        
        init();
    }
};