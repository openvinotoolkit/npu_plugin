#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(fuse_scale, case_conv)
{

    mv::OpModel om("testModel");

    auto input = om.input({64, 64, 16}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, {3, 3, 16, 32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    std::vector<double> scalesData = mv::utils::generateSequence<double>(32);
    auto scales = om.constant(scalesData, {32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, "biases");
    auto scale = om.scale(conv, scales);
    auto scaleOp = om.getSourceOp(scale);
    om.output(scale);
    
    auto outputOp = scaleOp.leftmostChild();
    
    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;

    mv::pass::PassRegistry::instance().find("FuseScale")->run(om, dummyTargDesc, dummyCompDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 3);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    mv::Tensor scaleParam("scale", {32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, scalesData);
    mv::Tensor originalWeights("originalWeights", {3, 3, 16, 32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, weightsData);
    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);

    for (unsigned i = 0; i < convOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getInputTensor(1)->getData()[i], newWeigths.getData()[i]);

}

TEST(fuse_scale, case_conv_bias_fused)
{

    mv::OpModel om("testModel");

    auto input = om.input({64, 64, 16}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, {3, 3, 16, 32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    std::vector<double> biasesData = mv::utils::generateSequence<double>(32);
    auto biases = om.constant(biasesData, {32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, "biases");
    auto bias = om.bias(conv, biases);
    std::vector<double> scalesData = mv::utils::generateSequence<double>(32);
    auto scales = om.constant(scalesData, {32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, "scales");
    auto scale = om.scale(bias, scales);
    auto scaleOp = om.getSourceOp(scale);
    om.output(scale);
    
    auto outputOp = scaleOp.leftmostChild();

    mv::json::Object dummyCompDesc;
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;

    mv::pass::PassRegistry::instance().find("FuseBias")->run(om, dummyTargDesc, dummyCompDesc, compOutput);
    mv::pass::PassRegistry::instance().find("FuseScale")->run(om, dummyTargDesc, dummyCompDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 4);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    mv::Tensor scaleParam("scale", {32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, scalesData);
    mv::Tensor originalWeights("originalWeights", {3, 3, 16, 32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, weightsData);
    mv::Tensor originalBiases("originalBiases", {32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor, biasesData);
    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);
    mv::Tensor newBiases = mv::math::multiply(originalBiases, scaleParam);

    for (unsigned i = 0; i < convOp->getInputTensor(1)->getData().size(); ++i)
        ASSERT_FLOAT_EQ(convOp->getInputTensor(1)->getData()[i], newWeigths.getData()[i]);

    auto biasVector = dm.findTensor(convOp->get<std::string>("bias"))->getData();
    for (unsigned i = 0; i < biasVector.size(); ++i)
        ASSERT_FLOAT_EQ(biasVector[i], newBiases.getData()[i]);

}

