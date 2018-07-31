#include "gtest/gtest.h"
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"
#include "include/mcm/computation/op/computation_op.hpp"
#include "include/mcm/computation/model/computation_element.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(jsonable, int)
{
    int number = -1;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "-1");
}

TEST(jsonable, unsigned_int)
{
    unsigned int number = 1;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "1");
}

TEST(jsonable, float)
{
    float number = 1.56;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "1.56");
}

TEST(jsonable, vector4d)
{
    mv::Vector4D<float> vec;
    vec.e0 = 1.0;
    vec.e1 = 2.0;
    vec.e2 = 3.0;
    vec.e3 = 4.0;
    mv::json::Value v = mv::Jsonable::toJsonValue(vec);
    std::string result(v.stringify());
    ASSERT_EQ(result, "[1.0,2.0,3.0,4.0]");
}

TEST(jsonable, bool)
{
    bool true_value = true;
    mv::json::Value v = mv::Jsonable::toJsonValue(true_value);
    ASSERT_EQ(v.stringify(), "true");
    bool true_value_bis = mv::Jsonable::constructBoolTypeFromJson(v);
    ASSERT_EQ(true_value, true_value_bis);
    bool false_value = false;
    mv::json::Value v1 = mv::Jsonable::toJsonValue(false_value);
    ASSERT_EQ(v1.stringify(), "false");
    bool false_value_bis = mv::Jsonable::constructBoolTypeFromJson(v1);
    ASSERT_EQ(false_value, false_value_bis);
}

TEST(jsonable, attribute1)
{
    mv::Attribute att(mv::AttrType::DTypeType, mv::DType::Float);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    ASSERT_EQ(result, "{\"attrType\":\"dtype\",\"content\":\"Float\"}");
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    //std::cout << result2 << std::endl;
    ASSERT_EQ(result, result2);
}

TEST(jsonable, attribute2)
{
    mv::Vector4D<float> vec;
    vec.e0 = 1.0;
    vec.e1 = 2.0;
    vec.e2 = 3.0;
    vec.e3 = 4.0;
    mv::Attribute att(mv::AttrType::FloatVec4DType, vec);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    std::cout << result << std::endl;
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    //std::cout << result2 << std::endl;
    ASSERT_EQ(result, result2);
}

TEST(jsonable, attribute_bool)
{
    mv::Attribute att(mv::AttrType::BoolType, true);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    //std::cout << result << std::endl;
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    //std::cout << result2 << std::endl;
    ASSERT_EQ(result, result2);
}

TEST(jsonable, shape)
{
    mv::Shape s(3, 3, 64, 100);
    mv::json::Value v = mv::Jsonable::toJsonValue(s);
    std::string result(v.stringify());
    //std::cout << result << std::endl;
    mv::Shape s1(v);
    mv::json::Value v1 = mv::Jsonable::toJsonValue(s1);
    std::string result1(v1.stringify());
    //std::cout << result1 << std::endl;
    ASSERT_EQ(result1, result);
}

TEST(jsonable, operation)
{
    mv::op::Add op("add_test");
    mv::json::Value v = mv::Jsonable::toJsonValue(op);
    std::string result(v.stringify());
    //std::cout << result << std::endl;

    mv::op::Add op2(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(op2);
    std::string result2(v2.stringify());

    ASSERT_EQ(result, result2);
}

TEST(jsonable, memory_allocator)
{
    mv::MemoryAllocator m("test_allocator", 2048);
    mv::Shape s(3, 3, 64);
    mv::Tensor t("test_tensor", s, mv::DType::Float, mv::Order::ColumnMajor);
    mv::Tensor t1("test_tensor1", s, mv::DType::Float, mv::Order::ColumnMajor);

    m.allocate(t, 0);
    m.allocate(t1, 0);

    mv::json::Value v = mv::Jsonable::toJsonValue(m);
    std::string result(v.stringify());
    std::cout << result << std::endl;
    ASSERT_EQ(result, "{\"max_size\":2048,\"name\":\"test_allocator\",\"states\":[{\"buffers\":[{\"layout\":\"plain\",\"lenght\":576,\"name\":\"test_tensor\",\"offset\":0},{\"layout\":\"plain\",\"lenght\":576,\"name\":\"test_tensor1\",\"offset\":576}],\"free_space\":896,\"stage\":0}]}");
}

TEST(jsonable, tensor)
{
    mv::Shape s(3, 3, 64);
    mv::Tensor t("test_tensor", s, mv::DType::Float, mv::Order::ColumnMajor);
    mv::json::Value v = mv::Jsonable::toJsonValue(t);
    std::string result(v.stringify());
    std::cout << result << std::endl;
    //ASSERT_EQ(result, "{\"attributes\":{\"dType\":{\"attrType\":\"dtype\",\"content\":\"Float\"},\"order\":{\"attrType\":\"order\",\"content\":\"LastDimMajor\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"shape\",\"content\":[3,3,64]}},\"name\":\"test_tensor\"}");
    mv::Tensor t1(v);
    mv::json::Value v1 = mv::Jsonable::toJsonValue(t1);
    std::string result1(v1.stringify());
    ASSERT_EQ(result, result1);
}



