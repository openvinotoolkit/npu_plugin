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
    double inf = std::numeric_limits<double>::infinity();

    auto input0 = om.input({64,64,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#9");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (1*1*16*16);
    auto weights0 = om.constantInt(weightsData0,{1,1,16,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.002022345317527652},{-0.2541584074497223},{0.26153966784477234}}, "conv/BiasAdd_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "conv/BiasAdd#10");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (16);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.5861533029237762e-05},{-inf},{inf}}, "conv/BiasAdd_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*16*64);
    auto weights1 = om.constantInt(weightsData1,{3,3,16,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.0029147472232580185},{-0.3575820028781891},{0.38567855954170227}}, "conv_1_weights#4");
    auto conv1 = om.conv(bias_c0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv_1#11");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.2860762328491546e-05},{-inf},{inf}}, "conv_1_bias#5");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*64*64);
    auto weights2 = om.constantInt(weightsData2,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.002709108404815197},{-0.34028318524360657},{0.3505394756793976}}, "output_weights#7");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "output#12");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.0623954949551262e-05},{-inf},{inf}}, "output_bias#8");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_c2);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
