#include <cmath>
#include "base.h"

struct PML_Node
{
    float idx;    
    float kappa_E;
    float kappa_M;
    float sigma_E;
    float sigma_M;
    float alpha_E;
    float alpha_M;
    float b_E;
    float c_E;
    float b_M;
    float c_M;        
};

class PML
{    
private:    
    float order;
    float kappa_max;
    float sigma_max;
    float alpha_min;
    float alpha_max;    
    void init_PML_part(PML_Node *PML_nodes, bool reverse_i, float idx_offset=0);       
    void init();

public: 
    int n_PML_nodes_per_part;
    PML_Node *part1;
    PML_Node *part2;

    PML(int n_PML_nodes_per_part_=20, float order_=3, float kappa_max_=2)
    {
        n_PML_nodes_per_part = n_PML_nodes_per_part_;        
        order = order_;        
        kappa_max = kappa_max_;
        sigma_max = (order + 1) / (150*M_PI*dx); // where this formulation comes from?
        printf("sigma_max: %f \n", sigma_max);
        alpha_min = 0;
        alpha_max = 4e-5;
        init();
    }    

    ~PML()
    {
        delete [] part1;
        delete [] part2;
    }
};

void PML::init_PML_part(PML_Node *PML_nodes, bool reverse_i, float idx_offset)
{
    vector<float> vec_idx(n_PML_nodes_per_part);
    for (int i=0; i<n_PML_nodes_per_part; i++)
    {
        if(reverse_i)
            vec_idx[i] = (float)(n_PML_nodes_per_part - i);
        else
            vec_idx[i] = (float)(i + 1.0);
    }

    for (int i=0; i<n_PML_nodes_per_part; ++i)
    {
        PML_Node node;
        node.idx = vec_idx[i] + idx_offset;        
        float p_E = powf((node.idx-0.25) / (float)n_PML_nodes_per_part, order);
        float p_M = powf((node.idx+0.25) / (float)n_PML_nodes_per_part, order);
        node.kappa_E = 1 + (kappa_max - 1) * p_E;
        node.kappa_M = 1 + (kappa_max - 1) * p_M;
        node.sigma_E = sigma_max * p_E;
        node.sigma_M = (mu0 / eps0) * sigma_max * p_M;
        //node.sigma_M = sigma_max * p_M;
        node.alpha_E  = alpha_min + (alpha_max-alpha_min)*(1 - p_E);
        node.alpha_M  = (mu0 / eps0) * (alpha_min + (alpha_max-alpha_min)*(1 - p_M));
        // node.alpha_E = 0;
        // node.alpha_M = 0;
        float temp = node.kappa_E*eps0 + dt*(node.kappa_E*node.alpha_E + node.sigma_E);
        node.b_E = (node.kappa_E * eps0) / temp;
        node.c_E = (dt * node.sigma_E) / (node.kappa_E * temp);
        temp = node.kappa_M*mu0 + dt*(node.kappa_M*node.alpha_M + node.sigma_M);
        node.b_M = (node.kappa_M * mu0) / temp;
        node.c_M = (dt * node.sigma_M) / (node.kappa_M * temp);

        PML_nodes[i] = node;

        /*
        bex = (e0*kappa_e)./(e0*kappa_e+dt.*(o_pex+alpha_e.*kappa_e));
        cex = (-dt*o_pex)./(dx*kappa_e.*(e0*kappa_e+dt.*(o_pex+alpha_e.*kappa_e)));
        bmx = (u0*kappa_m)./(u0*kappa_m+dt.*(o_pmx+alpha_m.*kappa_m));
        cmx = (-dt*o_pmx)./(dx*kappa_m.*(u0*kappa_m+dt.*(o_pmx+alpha_m.*kappa_m)));        
        */
    }
}

void PML::init()
{    
    part1 = new PML_Node[n_PML_nodes_per_part];          
    init_PML_part(part1, true, 0);    
    part2 = new PML_Node[n_PML_nodes_per_part];  
    init_PML_part(part2, false, 0);    
}

inline PML_Node *get_PML_node(PML_Node *part1, PML_Node *part2, int n_PML, int n, int node_i)
{               
    PML_Node *pml_node = NULL;    
    int part2_begin_i = n - n_PML;              
    if (node_i < n_PML)
    {                                        
        pml_node = part1 + node_i;
    }
    else if (node_i >= part2_begin_i)                                                                
    {
        pml_node = part2 + (node_i - part2_begin_i);
    }
    return pml_node;
}