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
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({56, 56, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*64*64);
    auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "res2a_branch2a_weights");
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}}, "res2a_branch2a#4");

    std::vector<int64_t> biasWeightsData = mv::utils::generateSequence<int64_t>(64);
    auto biasWeights = om.constantInt(biasWeightsData, {64}, mv::DType("UInt8"), mv::Order::getColMajorID(1));

    auto bias = om.bias(conv, biasWeights);

    om.output(bias);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng keembay_adapt_model.dot -o keembay_adapt_model.png");
    system("dot -Tpng dma_model.dot -o dma_model.png");
    system("dot -Tpng TransitiveReduction.dot -o TransitiveReduction.png");
    system("dot -Tpng deallocation_model_data.dot -o deallocation_model_data.png");
    system("dot -Tpng deallocation_model_control.dot -o deallocation_model_control.png");
    system("dot -Tpng DmaControlFlows_model.dot -o DmaControlFlows_model.png");
    system("dot -Tpng InputOutputControlFlows_model.dot -o InputOutputControlFlows_model.png");
    system("flatc -t ../../schema/graphfile/src/schema/graphfile.fbs -- blob.bin");
}
