#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include "include/mcm/logger/logger.hpp"

#include <iostream>
#include <fstream>

// This example demonstrates the DPUConvolution pass:
// which replaces all convolution operations with DPU tasks,
// and adds appropriate DMA tasks (for DDR-to-CMX and back),
// and de-allocation tasks for the temporary CMX buffers.

int main()
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);

    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});
    auto pool1 = om.maxPool(conv1, {2, 2}, {2, 2}, {0, 0, 0, 0});
    auto pool2 = om.maxPool(conv1, {4, 4}, {2, 2}, {1, 1, 1, 1});

    std::vector<double> weights3Data = mv::utils::generateSequence<double>(3*3*16*16);
    auto weights2 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv2 = om.conv(pool1, weights2, {1, 1}, {1, 1, 1, 1});

    auto weights3 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {1, 1, 1, 1});

    auto add1 = om.add({conv2, conv3});

    auto weights4 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv4 = om.conv(add1, weights4, {1, 1}, {1, 1, 1, 1});
    
    auto weights5 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));    
    auto conv5 = om.conv(conv4, weights5, {1, 1}, {1, 1, 1, 1});
    
    auto weights6 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv6 = om.conv(conv5, weights6, {1, 1}, {1, 1, 1, 1});
    
    // auto weights7 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    // auto conv7 = om.conv(conv6, weights7, {1, 1}, {1, 1, 1, 1});

    // auto weights8 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    // auto conv8 = om.conv(conv7, weights8, {1, 1}, {1, 1, 1, 1});

    // auto weights9 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    // auto conv9 = om.conv(conv8, weights9, {1, 1}, {1, 1, 1, 1});

    om.output(conv6);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
}
