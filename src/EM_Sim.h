#pragma once
#include "base.h"
#include "Base.h"
#include "Material.h"
#include "PML.h"

class EM_Sim : Base
{
private:    
protected:
    value_t *Hx;
    value_t *Hy;
    value_t *Ez;
    value_t *Q_E_x;
    value_t *Q_E_y;
    value_t *Q_M_x;
    value_t *Q_M_y;
    MaterialData materialData;
    bool use_pml;
    PML *pml_xdir;
    PML *pml_ydir;

public:
    EM_Sim(){}
    EM_Sim(value_t *Hx_, value_t *Hy_, value_t *Ez_, MaterialData &material_data_, bool use_pml_)
    {
        Q_E_x = NULL, Q_E_y = NULL, Q_M_x = NULL, Q_M_y = NULL;
        if (use_pml_)
        {
            int N = Nx * Ny;
            Q_E_x = new value_t[N];   
            Q_E_y = new value_t[N];   
            Q_M_x = new value_t[N];   
            Q_M_y = new value_t[N];   
            initialize_zero(Q_E_x, N);
            initialize_zero(Q_E_y, N);
            initialize_zero(Q_M_x, N);
            initialize_zero(Q_M_y, N);
        }
        Hx = Hx_;
        Hy = Hy_;
        Ez = Ez_;
        materialData = material_data_;
        use_pml = use_pml_;
        if (use_pml)
        {
            int n_PML = 30;
            int kappa_max = 10.0;
            pml_xdir = new PML(n_PML, 3.0, kappa_max);
            pml_ydir = new PML(n_PML, 3.0, kappa_max);
        }
    }

    ~EM_Sim()
    {
        delete pml_xdir;
        delete pml_ydir;
    }
    
    virtual void step_EM(int step_index) = 0;
};
