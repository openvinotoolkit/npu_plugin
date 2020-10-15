#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("ConvSoftmax");
    mv::OpModel& om = unit.model();

    auto input_shape = mv::Shape({1,1,100,1});
    auto weights_shape = mv::Shape({1,1,100,1000});

    // Load weights (u8 saved as int64)
    std::string weights_filename(mv::utils::projectRootPath() + "/tests/layer/conv_softmax/conv_softmax.w");
    std::vector<int64_t> weightsData = mv::utils::readWeightsFromFile<int64_t>(weights_filename);

    // Load biases (int32 saved as int64)
    std::string bias_filename(mv::utils::projectRootPath() + "/tests/layer/conv_softmax/conv_softmax.b");
    std::vector<int64_t> biasData = mv::utils::readWeightsFromFile<int64_t>(bias_filename);

    // Input
    auto input0 = om.input(input_shape, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#4");

    // Conv
    auto weights0 = om.constantInt(weightsData, weights_shape, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0033313746098428965},{-0.43268874287605286},{0.4168117642402649}}, "conv#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv#5");
    auto biasWeights0 = om.constantInt(biasData, {weights_shape[mv::IO_BATCH_DIMENSION]}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.6128427634830587e-05},{},{}}, "conv#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    // Softmax
    auto softmax0 = om.softmax(bias_c0, "C", mv::DType("Float16"), {{0},{0.00390625},{0.0},{0.99609375}}, "conv_softmax#6");

    // Output
    om.output(softmax0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
