#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/adaptation/fuse_passes.hpp"

TEST(fuse_scale, case_conv)
{

    mv::OpModel om;

    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    mv::dynamic_vector<mv::float_type> scalesData = mv::utils::generateSequence<mv::float_type>(32);
    auto scales = om.constant(scalesData, mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, "biases");
    auto scale = om.scale(conv, scales);
    auto scaleOp = om.getSourceOp(scale);
    om.output(scale);
    
    auto outputOp = scaleOp.leftmostChild();
    
    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;

    mv::pass::__fuse_pass_detail_::fuseScaleFcn(om, dummyTargDesc, dummyCompDesc);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 3);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    mv::Tensor scaleParam("scale", mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, scalesData);
    mv::Tensor originalWeights("originalWeights", mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, weightsData);
    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);

    for (unsigned i = 0; i < convOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getInputTensor(1)->getData()[i], newWeigths.getData()[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(convOp);

    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}

TEST(fuse_scale, case_conv_bias_fused)
{

    mv::OpModel om;

    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    mv::dynamic_vector<mv::float_type> biasesData = mv::utils::generateSequence<mv::float_type>(32);
    auto biases = om.constant(biasesData, mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, "biases");
    auto bias = om.bias(conv, biases);
    mv::dynamic_vector<mv::float_type> scalesData = mv::utils::generateSequence<mv::float_type>(32);
    auto scales = om.constant(scalesData, mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, "scales");
    auto scale = om.scale(bias, scales);
    auto scaleOp = om.getSourceOp(scale);
    om.output(scale);
    
    auto outputOp = scaleOp.leftmostChild();

    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;

    mv::pass::__fuse_pass_detail_::fuseBiasFcn(om, dummyTargDesc, dummyCompDesc);
    mv::pass::__fuse_pass_detail_::fuseScaleFcn(om, dummyTargDesc, dummyCompDesc);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 3);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    mv::Tensor scaleParam("scale", mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, scalesData);
    mv::Tensor originalWeights("originalWeights", mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, weightsData);
    mv::Tensor originalBiases("originalBiases", mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, biasesData);
    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);
    mv::Tensor newBiases = mv::math::multiply(originalBiases, scaleParam);

    for (unsigned i = 0; i < convOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getInputTensor(1)->getData()[i], newWeigths.getData()[i]);

    auto biasVector = convOp->getAttr("bias").getContent<mv::dynamic_vector<float>>();
    for (unsigned i = 0; i < biasVector.size(); ++i)
        ASSERT_FLOAT_EQ(biasVector[i], newBiases.getData()[i]);

    mv::ControlModel cm(om);
    mv::Control::OpDFSIterator cIt = cm.switchContext(convOp);

    ++cIt;
    ASSERT_EQ(*(outputOp), *cIt);

}

