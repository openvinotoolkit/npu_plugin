//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

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
    auto input0 = om.input({56,56,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#23");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights0 = om.constantInt(weightsData0,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.0028885104693472385},{-0.3315596878528595},{0.405010461807251}}, "res2a_branch1_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch1#24");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.265498369524721e-05},{-inf},{inf}}, "res2a_branch1_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights1 = om.constantInt(weightsData1,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0023823310621082783},{-0.2921203672885895},{0.31537407636642456}}, "res2a_branch2a_weights#4");
    auto conv1 = om.conv(input0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2a#25");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.8684950191527605e-05},{-inf},{inf}}, "res2a_branch2a_bias#5");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights2 = om.constantInt(weightsData2,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{138},{0.0030780788511037827},{-0.4242688715457916},{0.3606412410736084}}, "res2a_branch2b_weights#7");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2b#26");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2070897355442867e-05},{-inf},{inf}}, "res2a_branch2b_bias#8");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights3 = om.constantInt(weightsData3,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.003113032318651676},{-0.3866766393184662},{0.4071465730667114}}, "res2a_branch2c_weights#10");
    auto conv3 = om.conv(bias_c2, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a_branch2c#27");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2207969120936468e-05},{-inf},{inf}}, "res2a_branch2c_bias#11");
    auto bias_c3 = om.bias(conv3, biasWeights3, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise0 = om.add({bias_c0,bias_c3}, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2a/Relu#28");

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights4 = om.constantInt(weightsData4,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0026876654010266066},{-0.32668831944465637},{0.3586663603782654}}, "res2b_branch2a_weights#14");
    auto conv4 = om.conv(eltwise0, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2a#29");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0539863978920039e-05},{-inf},{inf}}, "res2b_branch2a_bias#15");
    auto bias_c4 = om.bias(conv4, biasWeights4, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights5 = om.constantInt(weightsData5,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0030561028979718685},{-0.389432817697525},{0.3898734152317047}}, "res2b_branch2b_weights#17");
    auto conv5 = om.conv(bias_c4, weights5, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2b#30");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1984717275481671e-05},{-inf},{inf}}, "res2b_branch2b_bias#18");
    auto bias_c5 = om.bias(conv5, biasWeights5, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (1*1*64*256);
    auto weights6 = om.constantInt(weightsData6,{1,1,64,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.0028176933992654085},{-0.3381056487560272},{0.38040614128112793}}, "res2b_branch2c_weights#20");
    auto conv6 = om.conv(bias_c5, weights6, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b_branch2c#31");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.1049777640437242e-05},{-inf},{inf}}, "res2b_branch2c_bias#21");
    auto bias_c6 = om.bias(conv6, biasWeights6, {{0},{0.003921568859368563},{0.0},{1.0}});

    auto eltwise1 = om.add({eltwise0,bias_c6}, {{0},{0.003921568859368563},{0.0},{1.0}}, "res2b/FakeQuantWithMinMaxArgs#32");

    om.output(eltwise1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
