//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

int main()
{
    mv::CompilationUnit unit("ConvSigmoid");
    mv::OpModel& om = unit.model();

    auto input_shape = mv::Shape({1,1,100,1});
    auto weights_shape = mv::Shape({1,1,100,1000});

    // Load weights (u8 saved as int64)
    std::string weights_filename(mv::utils::projectRootPath() + "/tests/layer/conv_sigmoid/conv_sigmoid.w");
    std::vector<int64_t> weightsData0 = mv::utils::readWeightsFromFile<int64_t>(weights_filename);

    // Load biases (int32 saved as int64)
    std::string bias_filename(mv::utils::projectRootPath() + "/tests/layer/conv_sigmoid/conv_sigmoid.b");
    std::vector<int64_t> biasWeightsData0 = mv::utils::readWeightsFromFile<int64_t>(bias_filename);

    // Input
    auto input0 = om.input({1,1,184,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#20");

    // FC
    auto weights0 = om.constantInt(weightsData0,{184,128}, mv::DType("UInt8"), mv::Order("WC"), {{128},{0.001087399199604988},{-0.13866370916366577},{0.13862307369709015}}, "fully_connected/BiasAdd_weights#1");
    auto fc0 = om.fullyConnected(input0, weights0, mv::DType("UInt8"), {{128},{0.0470588244497776},{-6.023529529571533},{5.976470470428467}}, "fully_connected/BiasAdd#21");
    auto biasWeights0 = om.constantInt(biasWeightsData0,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{8.528621037839912e-06},{},{}}, "fully_connected/BiasAdd_bias#2");
    auto bias_i0 = om.bias(fc0, biasWeights0, mv::DType("UInt8"), {{128},{0.0470588244497776},{-6.023529529571533},{5.976470470428467}});

    // Sigmoid
    auto sigmoid0 = om.sigmoid(bias_i0, mv::DType("UInt8"), {{0},{0.00390625},{0.0},{0.99609375}}, "fully_connected/Sigmoid#22");

    // Output
    om.output(sigmoid0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
