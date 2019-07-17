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
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#17");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (7*7*3*64);
    auto weights0 = om.constantInt(weightsData0,{7,7,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.002871257718652487},{-0.32948583364486694},{0.40268489718437195}}, "conv1_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {2, 3, 2, 3}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1#18");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.2519669073517434e-05},{-inf},{inf}}, "conv1_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.maxPool(bias_c0, {3, 3}, {2, 2}, {0, 1, 0, 1}, true, "", "floor", {{0},{0.003921568859368563},{0.0},{1.0}}, "pool1/max_pool#19");

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights1 = om.constantInt(weightsData1,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{137},{0.0030952668748795986},{-0.42331647872924805},{0.36597657203674316}}, "res2a_branch1_weights#5");
    auto conv1 = om.conv(pool0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch1#20");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2138301826780662e-05},{-inf},{inf}}, "res2a_branch1_bias#6");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights2 = om.constantInt(weightsData2,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{138},{0.0029387609101831913},{-0.404168039560318},{0.34521597623825073}}, "res2a_branch2a_weights#8");
    auto conv2 = om.conv(pool0, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2a#21");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1524552064656746e-05},{-inf},{inf}}, "res2a_branch2a_bias#9");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights3 = om.constantInt(weightsData3,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.0032645531464368105},{-0.43268874287605286},{0.3997723162174225}}, "res2a_branch2b_weights#11");
    auto conv3 = om.conv(bias_c2, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2b#22");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2802169294445775e-05},{-inf},{inf}}, "res2a_branch2b_bias#12");
    auto bias_c3 = om.bias(conv3, biasWeights3, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights4 = om.constantInt(weightsData4,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0032063836697489023},{-0.4181155264377594},{0.39951232075691223}}, "res2a_branch2c_weights#14");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2c#23");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2574053471325897e-05},{-inf},{inf}}, "res2a_branch2c_bias#15");
    auto bias_c4 = om.bias(conv4, biasWeights4, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise0 = om.add({bias_c1,bias_c4}, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a/FakeQuantWithMinMaxArgs#24");

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/eltwise_streaming.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
