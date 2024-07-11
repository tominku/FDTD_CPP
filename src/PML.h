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
    int n_PML_nodes_per_part;    
    float order;
    float kappa_max;
    float sigma_max;
    float alpha_min;
    float alpha_max;
    PML_Node *part1;
    PML_Node *part2;            
    void init();

public:    
    PML(int n_PML_nodes_per_part_, float order_, float kappa_max_=2)
    {
        n_PML_nodes_per_part = n_PML_nodes_per_part_;        
        order = order_;        
        kappa_max = kappa_max_;
        sigma_max = (order + 1) / (150*M_PI*dx); // where this formulation comes from?
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

void PML::init()
{    
    part1 = new PML_Node[n_PML_nodes_per_part];    
    part2 = new PML_Node[n_PML_nodes_per_part];        

    for (int i=0; i<n_PML_nodes_per_part; i++)
    {
        PML_Node node;
        node.idx = i;        
        float p = powf(node.idx, order);
        node.kappa_E = 1 + (kappa_max - 1) * p;
        node.kappa_M = node.kappa_E;
        node.sigma_E = sigma_max * p;
        node.sigma_M = (mu0 / eps0) * node.sigma_E;
        node.alpha_E  = alpha_min + (alpha_max-alpha_min)*(1 - p);
        node.alpha_M  = (mu0 / eps0) * node.alpha_E;
        float temp = node.kappa_E*eps0 + dt*(node.kappa_E*node.alpha_E + node.sigma_E);
        node.b_E = (node.kappa_E * eps0) / temp;
        node.c_E = (dt * node.sigma_E) / node.kappa_E * temp;
        temp = node.kappa_M*mu0 + dt*(node.kappa_M*node.alpha_M + node.sigma_M);
        node.b_M = (node.kappa_M * mu0) / temp;
        node.c_M = (dt * node.sigma_M) / node.kappa_M * temp;
        
        part1[i] = node;

        /*
        bex = (e0*kappa_e)./(e0*kappa_e+dt.*(o_pex+alpha_e.*kappa_e));
        cex = (-dt*o_pex)./(dx*kappa_e.*(e0*kappa_e+dt.*(o_pex+alpha_e.*kappa_e)));
        bmx = (u0*kappa_m)./(u0*kappa_m+dt.*(o_pmx+alpha_m.*kappa_m));
        cmx = (-dt*o_pmx)./(dx*kappa_m.*(u0*kappa_m+dt.*(o_pmx+alpha_m.*kappa_m)));        
        */
    }
}