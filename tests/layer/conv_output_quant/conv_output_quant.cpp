//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

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
    auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#170");
    //auto input0 = om.input({224,224,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-16319.999034851846},{16192.499042392066}}, "input#170");
    
    std::vector<int64_t> weightsData0 = read_weights_from_file<int64_t, uint8_t>(mv::utils::projectRootPath() + "/tests/layer/conv_output_quant/Relu6#0_weights#1.dat");
    auto weights0 = om.constantInt(weightsData0,{3,3,3,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{105},{0.002647720742970705},{-0.2793084979057312},{0.3958602845668793}}, "Conv/Relu6#0_weights#1");
    //auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "Conv/Relu6#171");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{1},{0.0},{1.0}}, "Conv/Relu6#171");

    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t, int32_t>(mv::utils::projectRootPath() + "/tests/layer/conv_output_quant/Relu6#0_bias#2.dat");
    auto biasWeights0 = om.constantInt(biasWeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.076643613690976e-05},{-inf},{inf}}, "Conv/Relu6#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    om.output(bias_c0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb_MC-Prefetch1.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("adapt", "PostTrainingQuantize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
