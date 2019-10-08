//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*48);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.04871978983283043},{-6.9027419090271},{5.472084999084473}}, "MobilenetV2/Conv/Relu6_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv/Relu6#171");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00038211600622162223},{-inf},{inf}}, "MobilenetV2/Conv/Relu6_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"),{{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> d_weightsData0 = mv::utils::generateSequence<int64_t> (3*3*48*1);
    auto d_weights0 = om.constantInt(d_weightsData0,{3,3,48,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{95},{0.4666302800178528},{-44.04890823364258},{74.47518157958984}}, "MobilenetV2/expanded_conv/depthwise/Relu6_weights#4");
    auto depthConv0 = om.depthwiseConv(bias_c0, d_weights0, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("UInt8"),{{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/expanded_conv/depthwise/Relu6#172");

    std::vector<int64_t> biasd_WeightsData0 = mv::utils::generateSequence<int64_t> (48);
    auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.010979099199175835},{-inf},{inf}}, "MobilenetV2/expanded_conv/depthwise/Relu6_bias#5");
    auto bias_cd0 = om.bias(depthConv0, biasdWeights0,mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*48*24);
    auto weights1 = om.constantInt(weightsData1,{1,1,48,24}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{167},{0.04477492719888687},{-7.413511753082275},{3.959320068359375}}, "MobilenetV2/expanded_conv/project/add_fold_weights#7");
    auto conv1 = om.conv(bias_cd0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"),{{126},{0.3486804664134979},{-43.933738708496094},{44.97977828979492}}, "MobilenetV2/expanded_conv/project/add_fold#173");
    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
