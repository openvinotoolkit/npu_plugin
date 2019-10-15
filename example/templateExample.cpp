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
    auto input0 = om.input({227,227,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*48);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{111},{0.0027499478310346603},{-0.30537644028663635},{0.3958602845668793}}, "conv1/conv1#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1/conv1#4");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.156821938115172e-05},{-inf},{inf}}, "conv1/conv1#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0,  {{0},{0.003921568859368563},{0.0},{1.0}});

    om.output(bias_c0);

    std::string compDescPath = "/home/johnbrady/git/mcmCompiler/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}