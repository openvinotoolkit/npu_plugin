#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

double inf = std::numeric_limits<double>::infinity();

template <typename T1, typename T2> std::vector<T1> read_weights_from_file(std::string input_file)
{
    std::string filePath = mv::utils::projectRootPath() + "/tests/layer/multiple_conv_mc_fp/" + input_file;
    std::ifstream file(filePath, std::ifstream::binary);
    T2 inputString;
    std::vector<T2> data;
    while(file.read(reinterpret_cast<char*>(&inputString), sizeof(T2)))
        data.push_back(inputString);
    file.close();
    std::vector<T1> return_data(data.begin(), data.end());
    return return_data;
}

static mv::Data::TensorIterator convBiasRelu(mv::OpModel& om,
                                        mv::Data::TensorIterator input,
                                        const mv::Shape& kernelShape,
                                        const std::string& id)
{

    std::vector<int64_t> weightsData = read_weights_from_file<int64_t, uint16_t>("weights_bias/fc_" + id +"_weights.dat");
    auto weights = om.constantInt("fc_" + id + "_weights", weightsData, kernelShape, mv::DType("Float16"), mv::Order::getColMajorID(4));
    auto conv = om.conv("fc_" + id, input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    weights->setQuantParams({{0},{1.0},{-inf},{inf}});
    conv->setQuantParams({{0},{1.0},{-inf},{inf}});

    auto biasShape = mv::Shape({kernelShape[3]});
    std::vector<int64_t> biasWeightsData = read_weights_from_file<int64_t, uint16_t>("weights_bias/fc_" + id +"_bias.dat");
    auto biasWeights = om.constantInt("fc_" + id + "_bias", biasWeightsData, biasShape, mv::DType("Float16"), mv::Order::getColMajorID(1));
    auto bias = om.bias("", conv, biasWeights);
    biasWeights->setQuantParams({{0},{1.0},{-inf},{inf}});
    bias->setQuantParams({{0},{1.0},{-inf},{inf}});

    auto relu = om.relu("fc_" + id + "_relu", bias);
    relu->setQuantParams({{0},{1.0},{-inf},{inf}});

    return relu;
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input = om.input("", {16,40,784,1}, mv::DType("Float16"), mv::Order::getZMajorID(4));
    input->setQuantParams({{0},{1.0},{-inf},{inf}});

    auto fc_0 = convBiasRelu(om, input, {1,1,784,1024},"0");

    auto fc_1 = convBiasRelu(om, fc_0, {1,1,1024,512}, "1");

    auto fc_2 = convBiasRelu(om, fc_1, {1,1,512,256}, "2");

    auto fc_3 = convBiasRelu(om, fc_2, {1,1,256,10}, "3");

    om.output("", fc_3);

    std::string compDescPath = mv::utils::projectRootPath() +
        "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
