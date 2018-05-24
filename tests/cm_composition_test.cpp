#include "gtest/gtest.h"
#include "include/fathom/computation/model/op_model.hpp"

TEST(computation_model, minimal_valid_composition)
{

    mv::OpModel om;
    // Check if empty model is invalid - should not be, because input and output is undefined
    ASSERT_FALSE(om.isValid());

    // Compose minimal valid computation model
    auto inIt = om.input(mv::Shape(1, 32, 32, 3), mv::DType::Float, mv::Order::NWHC);
    auto outIt = om.output(inIt);

    // Check if model is valid
    ASSERT_TRUE(om.isValid());

}

TEST(computation_model, minimal_functional_composition)
{

    mv::OpModel om;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto inIt = om.input(mv::Shape(1, 32, 32, 1), mv::DType::Float, mv::Order::NWHC);
    mv::vector<mv::float_type> weightsData =
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
     15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f};
    mv::ConstantTensor weights(mv::Shape(3, 3, 1, 3), mv::DType::Float, mv::Order::NWHC, weightsData);
    auto convIt = om.conv(inIt, weights, 4, 4, 1, 1);
    auto outIt = om.output(convIt);

    // Check if model is valid 
    ASSERT_TRUE(om.isValid());

    // Check output shape
    ASSERT_EQ(outIt->getOutputShape(), mv::Shape(1, 8, 8, 3));

    // Check number of convolution parameters
    ASSERT_EQ(convIt->attrsCount(), 10);

    // Check accessibility of convolution parameters
    ASSERT_EQ(convIt->getAttr("weights").getType(), mv::AttrType::TensorType);
    ASSERT_EQ(convIt->getAttr("strideX").getType(), mv::AttrType::ByteType);
    ASSERT_EQ(convIt->getAttr("strideY").getType(), mv::AttrType::ByteType);
    ASSERT_EQ(convIt->getAttr("padX").getType(), mv::AttrType::ByteType);
    ASSERT_EQ(convIt->getAttr("padY").getType(), mv::AttrType::ByteType);

    // Check parameters values
    ASSERT_EQ(convIt->getAttr("weights").getContent<mv::ConstantTensor>().getData(), weightsData);
    ASSERT_EQ(convIt->getAttr("strideX").getContent<mv::byte_type>(), 4);
    ASSERT_EQ(convIt->getAttr("strideY").getContent<mv::byte_type>(), 4);
    ASSERT_EQ(convIt->getAttr("padX").getContent<mv::byte_type>(), 1);
    ASSERT_EQ(convIt->getAttr("padY").getContent<mv::byte_type>(), 1);

}

TEST(computation_model, failure_sanity)
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseSilent);

    auto inIt = om.input(mv::Shape(1, 32, 32, 3), mv::DType::Float, mv::Order::NWHC);
    auto outIt = om.output(inIt);

    ASSERT_TRUE(om.addAttr(inIt, "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10)));
    ASSERT_FALSE(om.addAttr(inIt, "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10)));

}