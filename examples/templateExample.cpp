//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({64,64,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#6");

    // std::vector<int64_t> d_weightsData0 = mv::utils::generateSequence<int64_t> (1*1*3*1);
    // auto d_weights0 = om.constantInt(d_weightsData0,{1,1,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{70},{0.0008769026608206332},{-0.0611756406724453},{0.16243453323841095}}, "dw_conv1#0_weights#1");
    // auto depthConv0 = om.depthwiseConv(input0, d_weights0, {1, 1}, {0, 0, 0, 0}, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "dw_conv1#7");

    // std::vector<int64_t> biasd_WeightsData0 = mv::utils::generateSequence<int64_t> (3);
    // auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{3}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{6.877667601656867e-06},{-inf},{inf}}, "dw_conv1#0_bias#2");
    // auto bias_cd0 = om.bias(depthConv0, biasdWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*32);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{105},{0.002647720742970705},{-0.2793084979057312},{0.3958602845668793}}, "conv1/conv1#3_weights#4");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1/conv1#8");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.038321806845488e-05},{-inf},{inf}}, "conv1/conv1#3_bias#5");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});


    // Parallel branches
    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*32*16);
    auto weights1 = om.constantInt(weightsData1,{3,3,32,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{105},{0.002647720742970705},{-0.2793084979057312},{0.3958602845668793}}, "conv2a_weights");
    auto conv1 = om.conv(bias_c0, weights1, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv2a");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (16);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.038321806845488e-05},{-inf},{inf}}, "conv2a_bias");
    auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*32*16);
    auto weights2 = om.constantInt(weightsData2,{3,3,32,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{105},{0.002647720742970705},{-0.2793084979057312},{0.3958602845668793}}, "conv2b_weights");
    auto conv2 = om.conv(bias_c0, weights2, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv2b");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (16);
    auto biasWeights2 = om.constantInt(biasWeightsData2, {16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.038321806845488e-05},{-inf},{inf}}, "conv2b_bias");
    auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});


    om.output(bias_c1);
    om.output(bias_c2);

    std::string compDescPath =  mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-PrefetchAdaptive.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
