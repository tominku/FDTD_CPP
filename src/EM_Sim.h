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
    MaterialData materialData;
    bool use_pml;
    PML *pml_xdir;
    PML *pml_ydir;

public:
    EM_Sim(){}
    EM_Sim(value_t *Hx_, value_t *Hy_, value_t *Ez_, MaterialData &material_data_, bool use_pml_)
    {
        Hx = Hx_;
        Hy = Hy_;
        Ez = Ez_;
        materialData = material_data_;
        use_pml = use_pml_;
        if (use_pml)
        {
            pml_xdir = new PML(20, 3.0, 2.0);
            pml_ydir = new PML(20, 3.0, 2.0);
        }
    }
    
    virtual void step_EM(int step_index) = 0;
};
