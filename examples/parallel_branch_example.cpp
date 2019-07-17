//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    // std::string path = std::getenv("MDK_HOME");
    // double inf = std::numeric_limits<double>::infinity();

    // mv::CompilationUnit unit("parserModel");
    // mv::OpModel& om = unit.model();
    // auto input0 = om.input({7,7,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#14");

    // std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (1*1*3*1024);
    // auto weights0 = om.constantInt(weightsData0,{1,1,3,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{113},{0.0027890040073543787},{-0.315335750579834},{0.3958602845668793}}, "conv1_weights#1");
    // auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1#15");

    // std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (1024);
    // auto biasWeights0 = om.constantInt(biasWeightsData0,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.187454083468765e-05},{-inf},{inf}}, "conv1_bias#2");
    // auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    // std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*1024*512);
    // auto weights1 = om.constantInt(weightsData1,{1,1,1024,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.003638133406639099},{-0.46311089396476746},{0.4646131098270416}}, "res5a_branch1_weights#4");
    // auto conv1 = om.conv(bias_c0, weights1, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "res5a_branch1#16");

    // std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (512);
    // auto biasWeights1 = om.constantInt(biasWeightsData1,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.426718972652452e-05},{-inf},{inf}}, "res5a_branch1_bias#5");
    // auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.125490203499794},{0.0},{32.0}});

    // std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*1024*512);
    // auto weights2 = om.constantInt(weightsData2,{1,1,1024,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.0036103154998272657},{-0.4741959273815155},{0.44643452763557434}}, "res5a_branch2a_weights#7");
    // auto conv2 = om.conv(bias_c0, weights2, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.125490203499794},{0.0},{32.0}}, "res5a_branch2a#17");

    // std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (512);
    // auto biasWeights2 = om.constantInt(biasWeightsData2,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4158100384520367e-05},{-inf},{inf}}, "res5a_branch2a_bias#8");
    // auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.125490203499794},{0.0},{32.0}});

    // auto eltwise0 = om.add(bias_c1,bias_c2, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5a/Relu#18");

    // std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (1*1*512*512);
    // auto weights3 = om.constantInt(weightsData3,{1,1,512,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.003502847393974662},{-0.4360058009624481},{0.45722025632858276}}, "res5b_branch2a_weights#11");
    // auto conv3 = om.conv(eltwise0, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5b_branch2a#19");

    // std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (512);
    // auto biasWeights3 = om.constantInt(biasWeightsData3,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005769395502284169},{-inf},{inf}}, "res5b_branch2a_bias#12");
    // auto bias_c3 = om.bias(conv3, biasWeights3, {{0},{0.16470588743686676},{0.0},{42.0}});

    // auto eltwise1 = om.add(eltwise0,bias_c3, {{0},{0.16470588743686676},{0.0},{42.0}}, "res5b/FakeQuantWithMinMaxArgs#20");

    // om.output(eltwise1);

     std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({64,64,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    auto pool0 = om.maxPool(input0, {1, 1}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "pool/max_pool#4");

    auto pool1 = om.maxPool(input0, {1, 1}, {1, 1}, {0, 0, 0, 0}, true, "", "floor", {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "pool_1/max_pool#5");

    auto eltwise0 = om.add(pool0,pool1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "eltwise_test#6");

    om.output(eltwise0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/eltwise_streaming.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}