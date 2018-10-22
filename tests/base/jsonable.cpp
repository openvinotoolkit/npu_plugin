/*#include "gtest/gtest.h"
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

TEST(jsonable, unsigned)
{
    unsigned number = 1;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "1");
}

TEST(jsonable, double)
{
    double number = 1.56;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "1.56");
}

TEST(jsonable, vector4d)
{
    mv::Vector4D<double> vec;
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
    mv::Attribute att(mv::AttrType::DTypeType, mv::DTypeType::Float);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    ASSERT_EQ(result, "{\"attrType\":\"dtype\",\"content\":\"Float\"}");
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}

TEST(jsonable, attribute2)
{
    mv::Vector4D<double> vec;
    vec.e0 = 1.0;
    vec.e1 = 2.0;
    vec.e2 = 3.0;
    vec.e3 = 4.0;
    mv::Attribute att(mv::AttrType::FloatVec4DType, vec);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}

TEST(jsonable, attribute_bool)
{
    mv::Attribute att(mv::AttrType::BoolType, true);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}

TEST(jsonable, shape)
{
    mv::Shape s({3, 3, 64, 100});
    mv::json::Value v = mv::Jsonable::toJsonValue(s);
    std::string result(v.stringify());
    mv::Shape s1(v);
    mv::json::Value v1 = mv::Jsonable::toJsonValue(s1);
    std::string result1(v1.stringify());
    ASSERT_EQ(result1, result);
}

TEST(jsonable, operation)
{
    mv::op::Add op("add_test");
    mv::json::Value v = mv::Jsonable::toJsonValue(op);
    std::string result(v.stringify());
    mv::op::Add op2(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(op2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}

TEST(jsonable, memory_allocator)
{
    mv::MemoryAllocator m("test_allocator", 2048);
    mv::Shape s(3, 3, 64);
    mv::Tensor t("test_tensor", s, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    mv::Tensor t1("test_tensor1", s, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    m.allocate(t, 0);
    m.allocate(t1, 0);
    mv::json::Value v = mv::Jsonable::toJsonValue(m);
    std::string result(v.stringify());
    ASSERT_EQ(result, "{\"max_size\":2048,\"name\":\"test_allocator\",\"states\":[{\"buffers\":[{\"layout\":\"plain\",\"lenght\":576,\"name\":\"test_tensor\",\"offset\":0},{\"layout\":\"plain\",\"lenght\":576,\"name\":\"test_tensor1\",\"offset\":576}],\"free_space\":896,\"stage\":0}]}");
}

TEST(jsonable, tensor)
{
    mv::Shape s({3, 3, 64});
    mv::Tensor t("test_tensor", s, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    mv::json::Value v = mv::Jsonable::toJsonValue(t);
    std::string result(v.stringify());
    //ASSERT_EQ(result, "{\"attributes\":{\"dType\":{\"attrType\":\"dtype\",\"content\":\"Float\"},\"mv::Order\":{\"attrType\":\"mv::Order\",\"content\":\"LastDimMajor\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"shape\",\"content\":[3,3,64]}},\"name\":\"test_tensor\"}");
    mv::Tensor t1(v);
    mv::json::Value v1 = mv::Jsonable::toJsonValue(t1);
    std::string result1(v1.stringify());
    ASSERT_EQ(result, result1);
}


TEST(jsonable, computation_model)
{
    // Define blank computation model (op view)
    mv::OpModel om;

    // Initialize weights data
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> weights2Data = mv::utils::generateSequence<double>(5u * 5u * 8u * 16u);
    std::vector<double> weights3Data = mv::utils::generateSequence<double>(4u * 4u * 16u * 32u);

    // Compose model - use Composition API to create ops and obtain tensors
    auto input = om.input({128, 128, 3}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    auto weights1 = om.constant(weights1Data, {3, 3, 3, 8}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    auto conv1 = om.conv2D(input, weights1, {2, 2}, {1, 1, 1, 1});
    auto pool1 = om.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = om.constant(weights2Data, {5, 5, 8, 16}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    auto conv2 = om.conv2D(pool1, weights2, {2, 2}, {2, 2, 2, 2});
    auto pool2 = om.maxpool2D(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = om.constant(weights3Data, {4, 4, 16, 32}, mv::DTypeType::Float16, mv::Order(mv::Order::getColMajorID(3)));
    auto conv3 = om.conv2D(pool2, weights3, {1, 1}, {0, 0, 0, 0});
    om.output(conv3);

    // Obtain ops from tensors and add them to groups
    auto pool1Op = om.getSourceOp(pool1);
    auto pool2Op = om.getSourceOp(pool2);
    auto group1It = om.addGroup("pools");
    om.addGroupElement(pool1Op, group1It);
    om.addGroupElement(pool2Op, group1It);

    auto group2It = om.addGroup("convs");
    auto conv1Op = om.getSourceOp(conv1);
    auto conv2Op = om.getSourceOp(conv2);
    auto conv3Op = om.getSourceOp(conv3);
    om.addGroupElement(conv1Op, group2It);
    om.addGroupElement(conv2Op, group2It);
    om.addGroupElement(conv3Op, group2It);

    // Add groups to another group
    auto group3It = om.addGroup("ops");
    om.addGroupElement(group1It, group3It);
    om.addGroupElement(group2It, group3It);

    // Add ops that are already in some group to another group
    auto group4It = om.addGroup("first");
    om.addGroupElement(conv1Op, group4It);
    om.addGroupElement(pool1Op, group4It);

    mv::json::Value v = om.toJsonValue();
    mv::OpModel om2(v);
    mv::json::Value v2 = om2.toJsonValue();
    ASSERT_EQ(v.stringify(), v2.stringify());
}



*/