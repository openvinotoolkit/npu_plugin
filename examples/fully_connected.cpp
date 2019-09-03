#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

//NOTE: Does not work just for input size mismatch, op is actually ok
int main()
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("FullyConnected");
    mv::CompositionalModel& om = unit.model();

    auto input = om.input({1, 1, 2048, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}});
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(2048*1000);
    auto weights = om.constantInt(weightsData, {2048,1000}, mv::DType("UInt8"),  mv::Order("WH"), {{132},{0.0038084},{-0.49236152},{0.4787797}});
    auto fullyConnected = om.fullyConnected(input, weights, {{128},{0.00784314},{-1.00392163},{0.99607843}});

    std::vector<int64_t> biasWeightsData = mv::utils::generateSequence<int64_t>(1000);
    auto biasWeights = om.constantInt(biasWeightsData, {1000}, mv::DType("UInt8"), mv::Order::getColMajorID(1),{{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    auto bias = om.bias(fullyConnected, biasWeights, {{0},{2.98697796e-05},{-inf},{inf}});
    auto output = om.output(bias);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
