#include "gtest/gtest.h"
#include "include/fathom/computation/model/op_model.hpp"

TEST(computation_model, minimal_valid_composition)
{

    mv::OpModel om;
    // Check if empty model is invalid - should not be, because input and output is undefined
    ASSERT_FALSE(om.isValid());

    // Compose minimal valid computation model
    auto input = om.input(mv::Shape(32, 32, 3), mv::DType::Float, mv::Order::NWHC);
    om.output(input);

    // Check if model is valid
    ASSERT_TRUE(om.isValid());

}

TEST(computation_model, minimal_functional_composition)
{

    mv::OpModel om;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input = om.input(mv::Shape(32, 32, 1), mv::DType::Float, mv::Order::NWHC);
    /*float rawData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
     15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f};
    mv::dynamic_vector<mv::float_type> weightsData(rawData);*/

    float *rawData = new float[27]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
    15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f};
    mv::dynamic_vector<float> weightsData(rawData, 27);
    auto weights1 = om.constant(weightsData, mv::Shape(3, 3, 1, 3), mv::DType::Float, mv::Order::NWHC);
    auto conv = om.conv2D(input, weights1, {4, 4}, {1, 1, 1, 1});
    om.output(conv);

    // Check if model is valid 
    ASSERT_TRUE(om.isValid());

    // Check output shape
    //ASSERT_EQ(output->getOutputShape(), mv::Shape(1, 8, 8, 3));

    auto convOp = om.getSourceOp(conv);

    // Check number of convolution parameters
    ASSERT_EQ(convOp->attrsCount(), 10);

    // Check accessibility of convolution parameters
    ASSERT_EQ(convOp->getAttr("stride").getType(), mv::AttrType::UnsignedVec2DType);
    ASSERT_EQ(convOp->getAttr("padding").getType(), mv::AttrType::UnsignedVec4DType);

    // Check parameters values
    ASSERT_EQ(weights1->getData(), weightsData);
    ASSERT_EQ(convOp->getAttr("stride").getContent<mv::UnsignedVector2D>().e0, 4);
    ASSERT_EQ(convOp->getAttr("stride").getContent<mv::UnsignedVector2D>().e1, 4);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e0, 1);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e1, 1);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e2, 1);
    ASSERT_EQ(convOp->getAttr("padding").getContent<mv::UnsignedVector4D>().e3, 1);

}

TEST(computation_model, failure_sanity)
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseSilent);

    auto input = om.input(mv::Shape(32, 32, 3), mv::DType::Float, mv::Order::NWHC);
    om.output(input);

    ASSERT_TRUE(om.addAttr(om.getSourceOp(input), "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10)));
    ASSERT_FALSE(om.addAttr(om.getSourceOp(input), "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10)));

}