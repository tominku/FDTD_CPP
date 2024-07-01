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

class FileUtil : Base
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
        
        std::string className = toName();
        std::string msg = fmt::format("[{}] data_dir: {}", className, data_dir_path.c_str());
        print(msg);
        
        //auto output_file_path = data_dir_path / "output_cpu.txt";
        //std::cout << "[FileUtil] output_path: " <<output_file_path << std::endl;
    }

protected:
    std::string toName()
    {
        return "FileUtil";
    }

public:
    FileUtil()
    {        
        init();
    }
};