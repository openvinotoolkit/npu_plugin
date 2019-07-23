//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

template <typename T1, typename T2> std::vector<T1> read_weights_from_file(std::string input_file)
{
    std::ifstream file(input_file, std::ifstream::binary);
    T2 inputString;
    std::vector<T2> data;
    while(file.read(reinterpret_cast<char*>(&inputString), sizeof(T2)))
        data.push_back(inputString);
    file.close();
    std::vector<T1> return_data(data.begin(), data.end());
    return return_data;
}

int main()
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({56,56,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData0 = read_weights_from_file<int64_t, uint8_t>("/home/mmecchia/WORK/mcmCompiler/examples/weights_bias/res2a_branch2a_weights#1.dat");
    auto weights0 = om.constantInt(weightsData0,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.002728967694565654},{-0.332787424325943},{0.3630993366241455}}, "res2a_branch2a_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "res2a_branch2a#4");

    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t, int32_t>("/home/mmecchia/WORK/mcmCompiler/examples/weights_bias/res2a_branch2a_bias#2.dat");
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.1403668142738752e-05},{-inf},{inf}}, "res2a_branch2a_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_c0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
