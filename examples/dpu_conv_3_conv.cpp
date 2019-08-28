#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

// This example demonstrates the DPUConvolution pass:
// which replaces all convolution operations with DPU tasks,
// and adds appropriate DMA tasks (for DDR-to-CMX and back),
// and de-allocation tasks for the temporary CMX buffers.

int main()
{   
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({40, 40, 3, 1}, mv::DType("Float8"), mv::Order("NCHW"));

    std::vector<double> weightsData = mv::utils::generateSequence<double>(7*7*3*90);
    auto weights = om.constant(weightsData, {7, 7, 3, 90}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {3, 3, 3, 3});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(7*7*90*90);
    auto weights1 = om.constant(weightsData1, {7, 7, 90, 90}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {3, 3, 3, 3});

    std::vector<double> weightsData2 = mv::utils::generateSequence<double>(7*7*90*90);
    auto weights2 = om.constant(weightsData2, {7, 7, 90, 90}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv2 = om.conv(conv1, weights2, {1, 1}, {3, 3, 3, 3});

    om.output(conv2);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    //compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
