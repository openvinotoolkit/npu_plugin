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

    const int& getAttr1() const
    {
        return attr1;
    }

    StringEntry& setAttr2(int value)
    {
        attr2 = value;
        return *this;
    }

    const int& getAttr2() const
    {
        return attr2;
    }

};

class TestRegistry : public mv::Registry<TestRegistry, std::string, StringEntry>
{

};

namespace mv
{

    MV_DEFINE_REGISTRY(TestRegistry, std::string, StringEntry)

}

TEST(registry, add_element)
{
    
    MV_REGISTER_ENTRY(TestRegistry, std::string, StringEntry, "String1")
    .setAttr1(0)
    .setAttr2(1);

    MV_REGISTER_ENTRY(TestRegistry, std::string, StringEntry, "String2")
    .setAttr1(2)
    .setAttr2(3);
    
    ASSERT_EQ((mv::Registry<TestRegistry, std::string, StringEntry>::instance().size()), 2);
    ASSERT_FALSE((mv::Registry<TestRegistry, std::string, StringEntry>::instance().find("String1")) == nullptr);
    ASSERT_EQ((mv::Registry<TestRegistry, std::string, StringEntry>::instance().find("String1")->getAttr1()), 0);
    ASSERT_EQ((mv::Registry<TestRegistry, std::string, StringEntry>::instance().find("String1")->getAttr2()), 1);
    ASSERT_FALSE((mv::Registry<TestRegistry, std::string, StringEntry>::instance().find("String2")) == nullptr);
    ASSERT_EQ((mv::Registry<TestRegistry, std::string, StringEntry>::instance().find("String2")->getAttr1()), 2);
    ASSERT_EQ((mv::Registry<TestRegistry, std::string, StringEntry>::instance().find("String2")->getAttr2()), 3);
    ASSERT_EQ((mv::Registry<TestRegistry, std::string, StringEntry>::instance().find("String3")), nullptr);

}
