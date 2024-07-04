#include <omp.h>
#include "base.h"
#include "Material.h"
#include "EM_Sim.h"
#include "Config.h"

class EM_Probe
{
private:
    
public:
    std::string name;
    int total_steps;
    value_t *values;    
    int ix;
    int iy;
    float ratio_x;
    float ratio_y;
    float physical_x;
    float physical_y;

    EM_Probe(std::string &name_, int total_steps_, int ix_, int iy_)
    {
        name = name_;
        total_steps = total_steps_;
        values = new value_t[total_steps];
        initialize_zero(values, total_steps);
        ix = ix_;
        iy = iy_;

        //cout << "constructor " << name << endl;
    }

    ~EM_Probe()
    {
        //cout << "deleting values: " << name << endl;
        delete values;
    }
};

class EM_Probe_Manager
{
private:
    int total_steps;      
    std::vector<EM_Probe *> probes;  
public:            

    EM_Probe_Manager()
    {
        Config &config = Config::instance();    
        total_steps = config.total_steps;
        int num_probes = config.probes.size();        
        
        for (json &probe_json : config.probes)
        {
            std::string probe_name = probe_json["name"];            
            int ix = probe_json["ix"];
            int iy = probe_json["iy"];
            EM_Probe *probe = new EM_Probe(probe_name, total_steps, ix, iy);
            probes.push_back(probe);
        }         
    }

    void probe(value_t *Ez, int step)
    {
        int num_probes = probes.size();        
        Config &config = Config::instance();            
        int num_threads_ = config.num_threads;
        if (num_threads_ > num_probes)
            num_threads_ = num_probes;
        #pragma omp parallel for num_threads(num_threads_)
        for (int p=0; p<num_probes; ++p)
        {
            EM_Probe *probe = probes[p];
            int i = probe->ix;
            int j = probe->iy;
            int k = ij_to_k(i, j);
            value_t value = Ez[k];
            probe->values[step] = value;
        }
        //ij_to_k()
    }

    void save()
    {
        FileManager &fileManager = FileManager::instance();    
        auto path = fileManager.into_data_dir("em_prob.json");
        json j;
        // j["name"] = name;
        // std::vector<value_t> vec_values(values, values+total_steps);
        // j["values"] = vec_values;
        // fileManager.save_json(j, path);            
    }

    ~EM_Probe_Manager()
    {        
        //delete values;
    }
};