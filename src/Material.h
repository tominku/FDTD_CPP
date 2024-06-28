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

using namespace std;

class Material
{
private:
    string file;

public:
    Material(string file_)
    {
        file = file_;
    }

    void openFile()
    {
        struct passwd *pw = getpwuid(getuid());
        const char *c_homedir = pw->pw_dir;
        const string homedir = c_homedir;
        const string data_dir = homedir + "/.data/";
        
        cout << "material data_dir: " << data_dir << "\n";
        const string file_path = data_dir + file;
        std::cout << "material file path: " << file_path << std::endl;
        
        ifstream material_file(file_path.data());
        //cout << material_file.get();
        string temp;
        char buf[20];
        buf[19] = '\0';
        material_file.read(buf, 19);
        if( material_file.is_open() ){
            string line(buf);
            //material_file >> line;
            //getline(material_file, line);
            cout << "read:" << buf << endl;
            ///cout << line.length() << endl;
            //material_file.close();
        }
    }
};