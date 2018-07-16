#include "gtest/gtest.h"
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/utils/parser/json_text.hpp"

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

    float floatVal1 = 1.0f;
    float floatVal2 = 2.0f;
    mv::json::Object root;
    root["objectValue"]["floatValue1"] = floatVal1;
    root["floatValue2"] = floatVal2;
    ASSERT_EQ(root["objectValue"]["floatValue1"].get<float>(), floatVal1);

}

TEST(json, array_value)
{

    float floatVal1 = 1.0f;
    float floatVal2 = 2.0f;
    mv::json::Array array({floatVal1, floatVal2});
    mv::json::Object root;
    root["arrayValue"] = array;
    ASSERT_EQ(root["arrayValue"][0].get<float>(), floatVal1);
    ASSERT_EQ(root["arrayValue"][1].get<float>(), floatVal2);

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

TEST(json, copy_assign_object)
{

    int val1 = 1;
    float val1mod = 5.0f;

    mv::json::Object obj1;
    obj1["val1"] = val1;
    mv::json::Object obj2;
    obj2["val2"] = obj1;
    obj1["val1"] = val1mod;

    ASSERT_EQ(obj2["val2"]["val1"].get<int>(), val1);
    ASSERT_EQ(obj1["val1"].get<float>(), val1mod);

}

TEST(json, stringify_values)
{

    int intVal = 1;
    float floatVal = 1.0f;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Value nullVal;
    mv::json::Object objVal;
    mv::json::Array arrVal;

    std::string intStr = "1";
    std::string floatStr = "1.0";
    std::string strStr =  "\"" + strVal + "\"";
    std::string boolStr = "true";
    std::string nullStr = "null";
    std::string objStr = "{}";
    std::string arrStr = "[]";

    mv::json::Object root;
    root["floatValue"] = floatVal;
    root["intValue"] = intVal;
    root["strValue"] = strVal;
    root["boolValue"] = boolVal;
    root["nullValue"] = nullVal;
    root["objValue"] = objVal;
    root["arrValue"] = arrVal;

    ASSERT_EQ(root["floatValue"].stringify(), floatStr);
    ASSERT_EQ(root["intValue"].stringify(), intStr);
    ASSERT_EQ(root["strValue"].stringify(), strStr);
    ASSERT_EQ(root["boolValue"].stringify(), boolStr);
    ASSERT_EQ(root["nullValue"].stringify(), nullStr);
    ASSERT_EQ(root["objValue"].stringify(), objStr);
    ASSERT_EQ(root["arrValue"].stringify(), arrStr);

}

TEST(json, stringify_array)
{

    int intVal = 1;
    float floatVal = 1.0f;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Value nullVal;
    mv::json::Object objVal;
    mv::json::Array arrVal(
        {intVal, floatVal, strVal, boolVal, nullVal, objVal}
    );

    mv::json::Object root;
    root["arrayValue"] = arrVal;
    std::string arrStr = "[1,1.0,\"str\",true,null,{}]";
    ASSERT_EQ(root["arrayValue"].stringify(), arrStr);

}

TEST(json, stringify_object)
{

    int intVal = 1;
    float floatVal = 1.0f;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Value nullVal;
    mv::json::Array arrVal;
    
    mv::json::Object root = {
        {"intValue", intVal},
        {"floatValue", floatVal},
        {"strValue", strVal},
        {"boolValue", boolVal},
        {"nullValue", nullVal},
        {"arrValue", arrVal}
    };
    std::string objStr = "{\"arrValue\":[],\"nullValue\":null,\"boolValue\":true,\"strValue\":\"str\",\"floatValue\":1.0,\"intValue\":1}";
    //ASSERT_EQ(root.stringify(), objStr);

}

TEST(json, parser_text)
{

    std::string fileName = "./test.txt";
    mv::JSONTextParser parser(8);
    mv::json::Value obj;
    parser.parseFile(fileName, obj);

    //std::cout << obj.stringify() << std::endl;

}