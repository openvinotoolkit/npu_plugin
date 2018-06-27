#include "gtest/gtest.h"
#include "include/mcm/base/json/object.hpp"

TEST(json, object)
{

    mv::json::Object obj;
    int intVal = 1;
    float floatVal = 2.0f;
    ASSERT_TRUE(obj.emplace("intMember", intVal));
    ASSERT_TRUE(obj.emplace("floatMember", floatVal));
    ASSERT_EQ(obj["intMember"].get<int>(), intVal);
    ASSERT_EQ(obj["floatMember"].get<float>(), floatVal);
    //std::cout << std::is_convertible<std::string, mv::json::NumberInteger>::value << std::endl;
}
