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
    auto input0 = om.input({56,56,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#9");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights0 = om.constantInt(weightsData0,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.0026864642277359962},{-0.35103046894073486},{0.33401790261268616}}, "conv/BiasAdd_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "conv/BiasAdd#10");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.1070307411719114e-05},{-inf},{inf}}, "conv/BiasAdd_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*64*64);
    auto weights1 = om.constantInt(weightsData1,{3,3,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.0029504322446882725},{-0.386017769575119},{0.3663424551486969}}, "conv_1/BiasAdd_weights#4");
    auto conv1 = om.conv(bias_c0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "conv_1/BiasAdd#11");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.3140644771046937e-05},{-inf},{inf}}, "conv_1/BiasAdd_bias#5");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights2 = om.constantInt(weightsData2,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.0026795354206115007},{-0.35402852296829224},{0.3292529881000519}}, "output_weights#7");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "output#12");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.1015963284298778e-05},{-inf},{inf}}, "output_bias#8");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_c2);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
