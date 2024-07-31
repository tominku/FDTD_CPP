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

    EM_Probe_Manager()
    {
        Config &config = Config::instance();    
        total_steps = config.total_steps;
        int num_probes = config.probes.size();        
        
        // Tx Probe
        std::string tx_name = "Tx";
        EM_Probe *tx_probe = new EM_Probe(tx_name, total_steps, 0, 0);
        probes.push_back(tx_probe);

        // Rx Probe
        std::string rx_name = "Rx";
        EM_Probe *rx_probe = new EM_Probe(rx_name,
            total_steps, source_x+2, source_y+2);
        probes.push_back(rx_probe);

        // Custom Probe
        for (json &probe_json : config.probes)
        {
            std::string probe_name = probe_json["name"];            
            int ix = probe_json["ix"];
            int iy = probe_json["iy"];
            EM_Probe *probe = new EM_Probe(probe_name, total_steps, ix, iy);
            probes.push_back(probe);
        }         
    }

public:            
    static EM_Probe_Manager& instance()
    {
        static EM_Probe_Manager INSTANCE;
        return INSTANCE;
    }

    void probe_Tx(value_t value, int step)
    {
        EM_Probe *probe = probes[0];
        probe->values[step] = value;
    }

    void probe(value_t *Ez, int step)
    {
        int num_probes = probes.size();        
        // Config &config = Config::instance();            
        // int num_threads_ = config.num_threads;
        // if (num_threads_ > num_probes)
        //     num_threads_ = num_probes;
        //#pragma omp parallel for num_threads(num_threads_)
        for (int p=1; p<num_probes; ++p)
        {
            EM_Probe *probe = probes[p];
            int i = probe->ix;
            int j = probe->iy;
            int k = ij_to_k(i, j);
            value_t value = Ez[k];
            probe->values[step] = value;
        }        
    }

    void save()
    {        
        //FileManager &fileManager = FileManager::instance();    
        //std::string path = fileManager.into_data_dir("EM_probe.json");   
        
        json j_parent;
        //Config &config = Config::instance();
        //config.  
        j_parent["dt"] = dt;
        j_parent["probes"] = json::array();
        for (EM_Probe *probe : probes)
        {                        
            json j;
            std::string probe_name = probe->name;
            j["name"] = probe_name;
            j["ix"] = probe->ix;
            j["iy"] = probe->iy;
            j["total_steps"] = probe->total_steps;
            std::vector<float> data(probe->values, probe->values + total_steps);
            j["data"] = data;
            j_parent["probes"].push_back(j);            
        }            
        // j["name"] = name;
        // std::vector<value_t> vec_values(values, values+total_steps);
        // j["values"] = vec_values;
        //fileManager.save_json(j_parent, path);            
        Config &config = Config::instance();            
        config.save_json_to_output_dir(j_parent, "EM_probe.json");        
    }

    ~EM_Probe_Manager()
    {        
        for (EM_Probe *probe : probes)
        {
            delete probe;
        }
    }
};