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

#define ADD_CF 0 // 1=add control flow, 0=don't

int main()
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({225, 225, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3);
    auto weights = om.constant(weightsData, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {2, 2}, {0, 0, 0, 0});
    om.output(conv);

#if ADD_CF
    mv::ControlModel cm(om);

    auto inputOp = om.getSourceOp(input);
    auto weightsOp = om.getSourceOp(weights);
    auto convOp = om.getSourceOp(conv);
    auto outputOp = om.getOp("Output_0");

    cm.defineFlow(inputOp, convOp);
    cm.defineFlow(weightsOp, convOp);
    cm.defineFlow(convOp, outputOp);
#endif

    std::string outputName("dpu_conv");

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
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

    system("dot -Tsvg dpu_conv.dot -o dpu_conv.png");
    system("dot -Tsvg dpu_conv_adapt.dot -o dpu_conv_adapt.png");
}
