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
#define KERNEL_SIZE 128
#define ORIG_OUTPUT_CHANNELS 64
#define NUM_SPLITS 16
#define OUTPUT_CHANNELS (ORIG_OUTPUT_CHANNELS/NUM_SPLITS)
int main()
{
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({164, 164, 3, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");
    //original weights
    //std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(KERNEL_SIZE*KERNEL_SIZE*3*OUTPUT_CHANNELS);
    //auto weights = om.constantInt(weightsData, {KERNEL_SIZE, KERNEL_SIZE, 3, OUTPUT_CHANNELS}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "weights");
    //auto conv = om.conv(input, weights, {1, 1}, {KERNEL_SIZE/2, KERNEL_SIZE/2, KERNEL_SIZE/2, KERNEL_SIZE/2}, 1, 1, {{},{},{},{}}, "conv_1");
    //auto output = om.output(conv);

    //spliting into 2 weights and 2 conv
    std::vector<mv::Data::TensorIterator> convs(NUM_SPLITS);

    for (size_t i=0; i < NUM_SPLITS; i++)
    {
        std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(KERNEL_SIZE*KERNEL_SIZE*3*OUTPUT_CHANNELS);
        auto weights = om.constantInt(weightsData, {KERNEL_SIZE, KERNEL_SIZE, 3, OUTPUT_CHANNELS}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "weightsssweightsss"+i);
        std::string name = "conv_"+std::to_string(i);
        convs[i] = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}}, name);
    }
    auto concat = om.concat(convs);
    auto output = om.output(concat);
    auto outp = om.getOutput();
    std::cout << (*outp->getInputTensor(0)).getShape().toString() << std::endl;

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    //compDesc.remove("finalize", "RemoveDeallocationTasks"); //TODO remove

    // run only the passes to build the task graph
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
    //system("flatc -t ../../schema/graphfile/src/schema/graphfile.fbs -- blob.bin");
    std::cout << " DONE !! " << std::endl;
}
