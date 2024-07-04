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

    EM_Probe(std::string &name_, int total_steps_)
    {
        name = name_;
        total_steps = total_steps_;
        values = new value_t[total_steps];
        initialize_zero(values, total_steps);

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
            EM_Probe *probe = new EM_Probe(probe_name, total_steps);
            probes.push_back(probe);
        }         
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