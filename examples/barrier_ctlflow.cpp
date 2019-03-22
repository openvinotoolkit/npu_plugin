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
    mv::ControlModel cm(om);

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});
    auto pool1 = om.maxPool(conv1, {2, 2}, {2, 2}, {0, 0, 0, 0});
    auto pool2 = om.maxPool(conv1, {4, 4}, {2, 2}, {1, 1, 1, 1});

    auto add1 = om.add(pool1, pool2);

    std::vector<double> weights3Data = mv::utils::generateSequence<double>(3*3*16*32);
    auto weights3 = om.constant(weights3Data, {3, 3, 16, 32}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(add1, weights3, {1, 1}, {1, 1, 1, 1});

    om.output(conv3);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("adapt", "GenerateWeightsTables");
    unit.compilationDescriptor().remove("adapt", "InsertBarrierTasks");
    unit.compilationDescriptor().remove("finalize");
    unit.compilationDescriptor().remove("serialize");

    // run the passes to build the task graph
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");

    // for testing, add edge from partial serialization to task graph
    auto inbounddmaOp = om.getOp("DMATask_3");
    auto aconvOp = om.getOp("DPU_Conv_0");
    auto bconvOp = om.getOp("DMATask_2");
    cm.defineFlow(aconvOp, inbounddmaOp);

    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.compilationDescriptor().remove("adapt", "ConvertToTaskGraph");
    unit.compilationDescriptor().remove("adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("adapt", "GenerateWeightsTables");

    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o final_model.png");
}
