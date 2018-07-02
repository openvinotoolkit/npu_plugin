#include "gtest/gtest.h"
#include "include/mcm/base/json/object.hpp"
#include "include/mcm/base/json/value.hpp"

TEST(json, root)
{
    
    mv::json::Object root;
    
}

TEST(json, float_value)
{

    float floatVal = 1.0f;
    mv::json::Object root;
    root["floatValue"] = floatVal;
    ASSERT_FLOAT_EQ(root["floatValue"].get<float>(), floatVal);
    ASSERT_ANY_THROW(root["floatValue"].get<int>());
    ASSERT_ANY_THROW(root["floatValue"].get<std::string>());
    ASSERT_ANY_THROW(root["floatValue"].get<bool>());

}

TEST(json, int_value)
{

    int intVal = 1;
    mv::json::Object root;
    root["intValue"] = intVal;
    ASSERT_FLOAT_EQ(root["intValue"].get<int>(), intVal);
    ASSERT_ANY_THROW(root["intValue"].get<float>());
    ASSERT_ANY_THROW(root["intValue"].get<std::string>());
    ASSERT_ANY_THROW(root["intValue"].get<bool>());

}

TEST(json, string_value)
{

    std::string strVal = "str";
    mv::json::Object root;
    root["strValue"] = strVal;
    ASSERT_EQ(root["strValue"].get<std::string>(), strVal);
    ASSERT_ANY_THROW(root["strValue"].get<float>());
    ASSERT_ANY_THROW(root["strValue"].get<int>());
    ASSERT_ANY_THROW(root["strValue"].get<bool>());

}

TEST(json, bool_value)
{

    bool boolVal = true;
    mv::json::Object root;
    root["boolValue"] = boolVal;
    ASSERT_EQ(root["boolValue"].get<bool>(), boolVal);
    ASSERT_ANY_THROW(root["boolValue"].get<float>());
    ASSERT_ANY_THROW(root["boolValue"].get<int>());
    ASSERT_ANY_THROW(root["boolValue"].get<std::string>());

}

TEST(json, object_value)
{

    mv::json::Object root;
    root["objectValue"]["floatValue"] = 1.0f;

}

TEST(json, reassign_value_type)
{

    float floatVal = 1.0f;
    int intVal = 1;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Object root;

    root["value"] = floatVal;
    ASSERT_EQ(root["value"].get<float>(), floatVal);
    ASSERT_EQ(root.size(), 1);

    root["value"] = intVal;
    ASSERT_EQ(root["value"].get<int>(), intVal);
    ASSERT_EQ(root.size(), 1);

    root["value"] = strVal;
    ASSERT_EQ(root["value"].get<std::string>(), strVal);
    ASSERT_EQ(root.size(), 1);

    root["value"] = boolVal;
    ASSERT_EQ(root["value"].get<bool>(), boolVal);
    ASSERT_EQ(root.size(), 1);

}

TEST(json, stringify)
{

    float floatVal = 1.0f;
    std::string floatStr = "1.0";
    int intVal = 1;
    std::string intStr = "1";
    std::string strVal = "str";
    std::string strStr =  "\"" + strVal + "\"";
    bool boolVal = true;
    std::string boolStr = "true";
    mv::json::JSON root;

    root["floatValue"] = floatVal;
    ASSERT_EQ(root["floatValue"].stringify(), floatStr);
    root["intValue"] = intVal;
    ASSERT_EQ(root["intValue"].stringify(), intStr);
    root["strValue"] = strVal;
    ASSERT_EQ(root["strValue"].stringify(), strStr);
    root["boolValue"] = boolVal;
    ASSERT_EQ(root["boolValue"].stringify(), boolStr);

}