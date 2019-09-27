#include "gtest/gtest.h"
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/utils/parser/json_text.hpp"
#include "include/mcm/utils/env_loader.hpp"

TEST(json, double_value)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    double x = 1.0;
    mv::json::Value v = x;
    ASSERT_FLOAT_EQ(v.get<double>(), x);
    ASSERT_ANY_THROW(v.get<long long>());
    ASSERT_ANY_THROW(v.get<std::string>());
    ASSERT_ANY_THROW(v.get<bool>());
    ASSERT_ANY_THROW(v.get<mv::json::Array>());
    ASSERT_ANY_THROW(v.get<mv::json::Object>());
    ASSERT_ANY_THROW(v.get<mv::json::Null>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(json, int_value)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    long long x = 1;
    mv::json::Value v = x;
    ASSERT_EQ(v.get<long long>(), x);
    ASSERT_ANY_THROW(v.get<double>());
    ASSERT_ANY_THROW(v.get<std::string>());
    ASSERT_ANY_THROW(v.get<bool>());
    ASSERT_ANY_THROW(v.get<mv::json::Array>());
    ASSERT_ANY_THROW(v.get<mv::json::Object>());
    ASSERT_ANY_THROW(v.get<mv::json::Null>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(json, string_value)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    std::string x = "str";
    mv::json::Value v = x;
    ASSERT_EQ(v.get<std::string>(), x);
    ASSERT_ANY_THROW(v.get<double>());
    ASSERT_ANY_THROW(v.get<long long>());
    ASSERT_ANY_THROW(v.get<bool>());
    ASSERT_ANY_THROW(v.get<mv::json::Array>());
    ASSERT_ANY_THROW(v.get<mv::json::Object>());
    ASSERT_ANY_THROW(v.get<mv::json::Null>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(json, bool_value)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    bool x = true;
    mv::json::Value v = x;
    ASSERT_EQ(v.get<bool>(), x);
    ASSERT_ANY_THROW(v.get<double>());
    ASSERT_ANY_THROW(v.get<long long>());
    ASSERT_ANY_THROW(v.get<std::string>());
    ASSERT_ANY_THROW(v.get<mv::json::Array>());
    ASSERT_ANY_THROW(v.get<mv::json::Object>());
    ASSERT_ANY_THROW(v.get<mv::json::Null>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(json, object_value)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    mv::json::Object x;
    x["a1"] = 1.0;
    mv::json::Value v = x;
    ASSERT_EQ(v.get<mv::json::Object>(), x);
    ASSERT_ANY_THROW(v.get<double>());
    ASSERT_ANY_THROW(v.get<long long>());
    ASSERT_ANY_THROW(v.get<std::string>());
    ASSERT_ANY_THROW(v.get<mv::json::Array>());
    ASSERT_ANY_THROW(v.get<bool>());
    ASSERT_ANY_THROW(v.get<mv::json::Null>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(json, array_value)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    mv::json::Array x;
    x.append(1.0);
    mv::json::Value v = x;
    ASSERT_EQ(v.get<mv::json::Array>(), x);
    ASSERT_ANY_THROW(v.get<double>());
    ASSERT_ANY_THROW(v.get<long long>());
    ASSERT_ANY_THROW(v.get<std::string>());
    ASSERT_ANY_THROW(v.get<bool>());
    ASSERT_ANY_THROW(v.get<mv::json::Object>());
    ASSERT_ANY_THROW(v.get<mv::json::Null>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(json, null_value)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    mv::json::Value v;
    ASSERT_ANY_THROW(v.get<mv::json::Array>());
    ASSERT_ANY_THROW(v.get<double>());
    ASSERT_ANY_THROW(v.get<long long>());
    ASSERT_ANY_THROW(v.get<std::string>());
    ASSERT_ANY_THROW(v.get<bool>());
    ASSERT_ANY_THROW(v.get<mv::json::Object>());
    ASSERT_ANY_THROW(v.get<mv::json::Null>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(json, reassign_value_type)
{

    double doubleVal = 1.0f;
    long long intVal = 1;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Object root;

    root["value"] = doubleVal;
    ASSERT_EQ(root["value"].get<double>(), doubleVal);
    ASSERT_EQ(root.size(), 1);

    root["value"] = intVal;
    ASSERT_EQ(root["value"].get<long long>(), intVal);
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

    long long val1 = 1;
    double val1mod = 5.0f;

    mv::json::Object obj1;
    obj1["val1"] = val1;
    mv::json::Object obj2;
    obj2["val2"] = obj1;
    obj1["val1"] = val1mod;

    ASSERT_EQ(obj2["val2"]["val1"].get<long long>(), val1);
    ASSERT_EQ(obj1["val1"].get<double>(), val1mod);

}

TEST(json, stringify_values)
{

    long long intVal = 1;
    double doubleVal = 1.0f;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Value nullVal;
    mv::json::Object objVal;
    mv::json::Array arrVal;

    std::string intStr = "1";
    std::string doubleStr = "1.0";
    std::string strStr =  "\"" + strVal + "\"";
    std::string boolStr = "true";
    std::string nullStr = "null";
    std::string objStr = "{}";
    std::string arrStr = "[]";

    mv::json::Object root;
    root["doubleValue"] = doubleVal;
    root["intValue"] = intVal;
    root["strValue"] = strVal;
    root["boolValue"] = boolVal;
    root["nullValue"] = nullVal;
    root["objValue"] = objVal;
    root["arrValue"] = arrVal;

    ASSERT_EQ(root["doubleValue"].stringify(), doubleStr);
    ASSERT_EQ(root["intValue"].stringify(), intStr);
    ASSERT_EQ(root["strValue"].stringify(), strStr);
    ASSERT_EQ(root["boolValue"].stringify(), boolStr);
    ASSERT_EQ(root["nullValue"].stringify(), nullStr);
    ASSERT_EQ(root["objValue"].stringify(), objStr);
    ASSERT_EQ(root["arrValue"].stringify(), arrStr);

}

TEST(json, stringify_array)
{

    long long intVal = 1;
    double doubleVal = 1.0f;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Value nullVal;
    mv::json::Object objVal;
    mv::json::Array arrVal(
        {intVal, doubleVal, strVal, boolVal, nullVal, objVal}
    );

    mv::json::Object root;
    root["arrayValue"] = arrVal;
    std::string arrStr = "[1,1.0,\"str\",true,null,{}]";
    ASSERT_EQ(root["arrayValue"].stringify(), arrStr);

}

TEST(json, stringify_object)
{

    long long intVal = 1;
    double doubleVal = 1.0f;
    std::string strVal = "str";
    bool boolVal = true;
    mv::json::Value nullVal;
    mv::json::Array arrVal;
    
    mv::json::Object root = {
        {"intValue", intVal},
        {"doubleValue", doubleVal},
        {"strValue", strVal},
        {"boolValue", boolVal},
        {"nullValue", nullVal},
        {"arrValue", arrVal}
    };
    std::string objStr = "{\"arrValue\":[],\"boolValue\":true,\"doubleValue\":1.0,\"intValue\":1,\"nullValue\":null,\"strValue\":\"str\"}";
    ASSERT_EQ(root.stringify(), objStr);

}

TEST(json, parser_text)
{

    std::string fileName = mv::utils::projectRootPath() + "/tests/data/test_01.json";
    mv::JSONTextParser parser(8);
    mv::json::Value obj;
    parser.parseFile(fileName, obj);

}
