#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
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

    auto input = om.input({225, 225, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3);
    auto weights = om.constant(weightsData, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {2, 2}, {0, 0, 0, 0});
    auto pool = om.maxPool(conv, {3 , 3}, {1, 1}, {1, 1, 1, 1});
    om.output(pool);

    std::string outputName("dpu_conv");

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2480.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateDot", "output", std::string(outputName + ".dot"));
    compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    compDesc.setPassArg("GenerateDot", "html", true);
    compDesc.remove("serialize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng dpu_conv.dot -o dpu_conv.png");
    system("dot -Tpng dpu_conv_adapt.dot -o dpu_conv_adapt.png");
    system("dot -Tpng dpu_conv_final.dot -o dpu_conv_final.png");
}
