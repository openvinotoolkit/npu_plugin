#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(fuse_scale, case_conv)
{

    mv::OpModel om("testModel");

    auto input = om.input({64, 64, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, {3, 3, 16, 32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)),{{},{},{},{}}, "weights");
    auto conv = om.conv(input, weights, {1, 1}, {1, 1, 1, 1}, 1);
    auto convOp = om.getSourceOp(conv);
    std::vector<double> scalesData = mv::utils::generateSequence<double>(32);
    auto scales = om.constant(scalesData, {32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)),{{},{},{},{}}, "biases");
    auto scale = om.scale(conv, scales);
    auto scaleOp = om.getSourceOp(scale);
    om.output(scale);
    
    auto outputOp = scaleOp.leftmostChild();
    
    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseScale")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 3);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);

    mv::Tensor scaleParam("scale", {32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)), scalesData);
    mv::Tensor originalWeights("originalWeights", {3, 3, 16, 32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)), weightsData);
    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);

    auto originalWeightsTensor = convOp->getInputTensor(1)->getDoubleData();
    auto originalWeightsTensorSize = originalWeightsTensor.size();

    auto newWeightsTensor = newWeigths.getDoubleData();

    for (unsigned i = 0; i < originalWeightsTensorSize; ++i)
        ASSERT_FLOAT_EQ(originalWeightsTensor[i], newWeightsTensor[i]);

}

TEST(fuse_scale, case_conv_bias_fused)
{

    mv::OpModel om("testModel");

    auto input = om.input({64, 64, 16, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, {3, 3, 16, 32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)),{{},{},{},{}}, "weights");
    auto conv = om.conv(input, weights, {1, 1}, {1, 1, 1, 1}, 1);
    auto convOp = om.getSourceOp(conv);
    std::vector<double> biasesData = mv::utils::generateSequence<double>(32);
    auto biases = om.constant(biasesData, {32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)),{{},{},{},{}}, "biases");
    auto bias = om.bias(conv, biases);
    std::vector<double> scalesData = mv::utils::generateSequence<double>(32);
    auto scales = om.constant(scalesData, {32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)), {{},{},{},{}},"scales");
    auto scale = om.scale(bias, scales);
    auto scaleOp = om.getSourceOp(scale);
    om.output(scale);
    
    auto outputOp = scaleOp.leftmostChild();

    mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");

    mv::pass::PassRegistry::instance().find("FuseBias")->run(om, dummyTargDesc, dummyPassDesc, compOutput);
    mv::pass::PassRegistry::instance().find("FuseScale")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    // Check general model properties
    mv::DataModel dm(om);
    ASSERT_EQ(om.opsCount(), 4);
    ASSERT_EQ(dm.tensorsCount(), 4);

    // Check predecessing operation
    ASSERT_EQ(convOp.childrenSize(), 1);
    
    mv::Tensor scaleParam("scale", {32}, mv::DType("Float16"), mv::Order("W"), scalesData);
    mv::Tensor originalWeights("originalWeights", {3, 3, 16, 32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(4)), weightsData);
    mv::Tensor originalBiases("originalBiases", {32}, mv::DType("Float16"), mv::Order(mv::Order::getColMajorID(1)), biasesData);
    mv::Tensor newWeigths = mv::math::multiply(originalWeights, scaleParam);
    mv::Tensor newBiases = mv::math::multiply(originalBiases, scaleParam);

    auto originalWeightsData = convOp->getInputTensor(1)->getDoubleData();
    auto originalWeightsDataSize = originalWeightsData.size();

    auto newWeigthsData = newWeigths.getDoubleData();

    for (unsigned i = 0; i < originalWeightsDataSize; ++i)
        ASSERT_FLOAT_EQ(originalWeightsData[i], newWeigthsData[i]);

    auto biasVector = dm.getTensor(convOp->get<std::string>("bias"))->getDoubleData();
    auto newBiasesData = newBiases.getDoubleData();
    for (unsigned i = 0; i < biasVector.size(); ++i)
        ASSERT_FLOAT_EQ(biasVector[i], newBiasesData[i]);

}

