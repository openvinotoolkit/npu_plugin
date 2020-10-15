#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

double inf = std::numeric_limits<double>::infinity();

template <typename T1, typename T2> std::vector<T1> read_weights_from_file(std::string input_file)
{
    std::string filePath = mv::utils::projectRootPath() + "/tests/layer/multiple_conv_sc/" + input_file;
    std::ifstream file(filePath, std::ifstream::binary);
    T2 inputString;
    std::vector<T2> data;
    while(file.read(reinterpret_cast<char*>(&inputString), sizeof(T2)))
        data.push_back(inputString);
    file.close();
    std::vector<T1> return_data(data.begin(), data.end());
    return return_data;
}

static mv::Data::TensorIterator convBias(mv::OpModel& om,
                                        mv::Data::TensorIterator input,
                                        const mv::Shape& kernelShape,
                                        const mv::QuantizationParams& actQuantParams,
                                        const mv::QuantizationParams& wtQuantParams,
                                        const mv::QuantizationParams& biasQuantParams,
                                        const std::string& id)
{
    std::vector<int64_t> weightsData = read_weights_from_file<int64_t, uint8_t>("weights_bias/fc_" + id +"_weights.dat");
    auto weights = om.constantInt(weightsData, kernelShape, mv::DType("UInt8"), mv::Order::getColMajorID(4), wtQuantParams, "fc_" + id + "_weights");
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), actQuantParams, "fc_" + id);

    auto biasShape = mv::Shape({kernelShape[3]});
    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t, int32_t>("weights_bias/fc_" + id +"_bias.dat");
    auto biasWeights = om.constantInt(biasWeightsData0, biasShape, mv::DType("UInt8"), mv::Order::getColMajorID(1), biasQuantParams, "fc_" + id + "_bias");
    auto bias = om.bias(conv, biasWeights, mv::DType("UInt8"), actQuantParams);

    return bias;
}

int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16,20,784,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input");

    auto fc_0 = convBias(om, input, {1,1,784,1024},
                                    {{0},{0.003921568859368563},{0.0},{1.0}},
                                    {{127},{0.003638133406639099},{-0.46311089396476746},{0.4646131098270416}},
                                    {{0},{2.853437945304904e-05},{-inf},{inf}},
                                    "0");

    auto fc_1 = convBias(om, fc_0, {1,1,1024,512},
                                    {{0},{0.003921568859368563},{0.0},{1.0}},
                                    {{133},{0.003733013290911913},{-0.4963448643684387},{0.45557352900505066}},
                                    {{0},{1.4639267646998633e-05},{-inf},{inf}},
                                    "1");

    auto fc_2 = convBias(om, fc_1, {1,1,512,256},
                                    {{0},{0.003921568859368563},{0.0},{1.0}},
                                    {{134},{0.0036992442328482866},{-0.49393245577812195},{0.449374794960022}},
                                    {{0},{1.4506839761452284e-05},{-inf},{inf}},
                                    "2");

    auto fc_3 = convBias(om, fc_2, {1,1,256,10},
                                    {{0},{0.003921568859368563},{0.0},{1.0}},
                                    {{136},{0.002595535246655345},{-0.3531047999858856},{0.30875667929649353}},
                                    {{0},{1.0178569027630147e-05},{-inf},{inf}},
                                    "3");

    om.output(fc_3);

    std::string compDescPath = mv::utils::projectRootPath() +
        "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
