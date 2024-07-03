#pragma once
#include <chrono>
#include <iostream>
#include "Base.h"

using namespace std::chrono;

class Timer : Base
{
private:
    //std::chrono::_V2::system_clock::time_point begin_point;
    //std::chrono::_V2::system_clock::time_point end_point;
    high_resolution_clock::time_point begin_point;
    high_resolution_clock::time_point end_point;

    int elapsed_time_micro;
    float elapsed_time_ms;

public:
    Timer()
    {
        begin_point = high_resolution_clock::now();        
    }

    void begin()
    {
        begin_point = high_resolution_clock::now();        
    }

    float end(bool as_ms = true)
    {
        end_point = high_resolution_clock::now();
        auto duration_micro = duration_cast<microseconds>(end_point - begin_point);
        elapsed_time_micro = duration_micro.count();                
        elapsed_time_ms = elapsed_time_micro / 1000.0;
        if (as_ms)
            return elapsed_time_ms;
        else        
            return (float)elapsed_time_micro;
    }

    void print_elapsed_time(std::string str)
    {
        std::string msg = fmt::format("Elapsed Time: {} ms === {}", elapsed_time_ms, str);        
        print(msg);
    }

    std::string toName()
    {
        return "Timer";
    }

    
};