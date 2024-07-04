#pragma once
#include "base.h"
#include "Base.h"
#include "Material.h"

class EM_Sim : Base
{
private:    
protected:
    value_t *Hx;
    value_t *Hy;
    value_t *Ez;
    MaterialData materialData;

public:
    EM_Sim(){}
    EM_Sim(value_t *Hx_, value_t *Hy_, value_t *Ez_, MaterialData &material_data_)
    {
        Hx = Hx_;
        Hy = Hy_;
        Ez = Ez_;
        materialData = material_data_;
    }
    
    virtual void step_EM(int step_index) = 0;
};
