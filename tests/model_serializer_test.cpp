#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/control_model.hpp"
#include "include/fathom/deployer/serializer.hpp"

TEST (model_serializer, blob_output) 
{
    // define test compute model: 1 convolution 
    mv::OpModel test_cm ;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto inIt = test_cm.input(mv::Shape(1, 32, 32, 1), mv::DType::Float, mv::Order::NWHC);
    mv::vector<mv::float_type> weightsData =
    { 0.111f, 0.111f, 0.111f, 0.111f, 0.111f, 0.111f, 0.111f, 0.111f, 0.111f };
    mv::ConstantTensor weights(mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::NWHC, weightsData);
    auto convIt = test_cm.conv(inIt, weights, 4, 4, 0, 0);
    auto outIt = test_cm.output(convIt);

    // Check if model is valid 
    ASSERT_TRUE(test_cm.isValid());

    // Check output shape
    ASSERT_EQ((*outIt).getOutputShape(), mv::Shape(1, 8, 8, 1));

    // Check number of convolution parameters
    ASSERT_EQ((*convIt).attrsCount(), 10);

    // Check parameters values
    ASSERT_EQ((*convIt).getAttr("opType").getContent<mv::string>(), "conv");
    ASSERT_EQ((*convIt).getAttr("weights").getContent<mv::ConstantTensor>().getData(), weightsData);
    ASSERT_EQ((*convIt).getAttr("weights").getContent<mv::ConstantTensor>().getShape()[0], 3);
    ASSERT_EQ((*convIt).getAttr("weights").getContent<mv::ConstantTensor>().getShape()[1], 3);
    ASSERT_EQ((*convIt).getAttr("weights").getContent<mv::ConstantTensor>().getShape()[2], 1);
    ASSERT_EQ((*convIt).getAttr("weights").getContent<mv::ConstantTensor>().getShape()[3], 1);
    ASSERT_EQ((*convIt).getAttr("strideX").getContent<mv::byte_type>(), 4);
    ASSERT_EQ((*convIt).getAttr("strideY").getContent<mv::byte_type>(), 4);
    ASSERT_EQ((*convIt).getAttr("padX").getContent<mv::byte_type>(), 0);
    ASSERT_EQ((*convIt).getAttr("padY").getContent<mv::byte_type>(), 0);
    ASSERT_EQ((*convIt).getInputShape()[1], 32);    // X dim
    ASSERT_EQ((*convIt).getInputShape()[2], 32);    // Y dim
    ASSERT_EQ((*convIt).getInputShape()[3], 1);     // Z dim (aka C)
    ASSERT_EQ((*convIt).getOutputShape()[1], 8);    // X dim
    ASSERT_EQ((*convIt).getOutputShape()[2], 8);    // Y dim
    ASSERT_EQ((*convIt).getOutputShape()[3], 1);    // Z dim 

    mv::ControlModel cm(test_cm);

    // declare serializer as blob
    mv::Serializer gs(mv::mvblob_mode);

    // serialize compute model to file
    uint64_t filesize = gs.serialize(cm, "test_1conv.blob");

    // compare filesize written to expected
    EXPECT_EQ (692, filesize) << "ERROR: wrong blob size";

}
