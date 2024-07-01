#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define FMT_HEADER_ONLY
#include "fmt/format.h"

namespace fs = std::filesystem;

std::string className(const std::string& prettyFunction)
{
    size_t colons = prettyFunction.find("::");
    if (colons == std::string::npos)
        return "::";
    size_t begin = prettyFunction.substr(0,colons).rfind(" ") + 1;
    size_t n = colons - begin;

    return prettyFunction.substr(begin, n);
}

#define __CLASS_NAME__ className(__PRETTY_FUNCTION__)

class Base
{
private:

protected:
    
    Base()
    {
    
    }

    virtual std::string toName() = 0;

public:
    void print(std::string message)
    {
        std::cout << message << std::endl;
    }
};