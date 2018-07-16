#include "gtest/gtest.h"
#include "include/mcm/base/registry.hpp"

class StringEntry
{

    std::string name_;
    int attr1;
    int attr2;

public:

    StringEntry(const std::string& name) : 
    name_(name)
    {

    }

    StringEntry& setAttr1(int value)
    {
        attr1 = value;
        return *this;
    }

    StringEntry& setAttr2(int value)
    {
        attr2 = value;
        return *this;
    }
    

};

MV_DEFINE_REGISTRY(StringEntry)

TEST(registry, add_element)
{
    
    MV_REGISTER_ENTRY(StringEntry, String1)
    .setAttr1(0)
    .setAttr2(1);



}
