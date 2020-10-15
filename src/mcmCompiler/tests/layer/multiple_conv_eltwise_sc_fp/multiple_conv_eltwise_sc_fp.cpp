#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

double inf = std::numeric_limits<double>::infinity();

template <typename T1, typename T2> std::vector<T1> read_weights_from_file(std::string input_file)
{
    std::string filePath = mv::utils::projectRootPath() + "/tests/layer/multiple_conv_eltwise_sc_fp/" + input_file;
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
    auto weights = om.constantInt(weightsData, kernelShape, mv::DType("Float16"), mv::Order::getColMajorID(4), {{0},{1.0},{-inf},{inf}}, "fc_" + id + "_weights");
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "fc_" + id);

    auto biasShape = mv::Shape({kernelShape[3]});
    std::vector<int64_t> biasWeightsData = read_weights_from_file<int64_t, uint16_t>("weights_bias/fc_" + id +"_bias.dat");
    auto biasWeights = om.constantInt(biasWeightsData, biasShape, mv::DType("Float16"), mv::Order::getColMajorID(1), {{0},{1.0},{-inf},{inf}}, "fc_" + id + "_bias");
    auto bias = om.bias(conv, biasWeights, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}});

    auto relu = om.relu(bias, mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "fc_" + id + "_relu");

    return relu;
}

static mv::Data::TensorIterator add(mv::OpModel& om,
                                        std::vector<mv::Data::TensorIterator> inputs,
                                        const std::string& id)
{
    auto add = om.eltwise(inputs, "Add", mv::DType("Float16"), {{0},{1.0},{-inf},{inf}}, "add_" + id);
    return add;
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16,10,784,1}, mv::DType("Float16"), mv::Order::getZMajorID(4), {{0},{1.0},{-inf},{inf}}, "input");

    auto fc_0 = convBiasRelu(om, input, {1,1,784,256}, "0");
    auto fc_1 = convBiasRelu(om, input, {1,1,784,256}, "1");
    auto add_2 = add(om, {fc_0, fc_1}, "2");
    auto fc_3 = convBiasRelu(om, add_2, {1,1,256,10}, "3");

    om.output(fc_3);

    std::string compDescPath = mv::utils::projectRootPath() +
        "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
