#include "gtest/gtest.h"
#include "include/mcm/base/json/object.hpp"

TEST(json, object)
{
    
    mv::json::Object obj;
    int intVal = 1;
    float floatVal = 2.0f;

    obj.emplace("floatVal", floatVal);
    float a = obj["floatVal"].get<float>();
    std::cout << a << std::endl;
}
