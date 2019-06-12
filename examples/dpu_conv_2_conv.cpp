#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "meta/include/mcm/op_model.hpp"

#include "iostream"
#include "fstream"

int main()
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({56,56,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#9");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*64);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.0021500778384506702},{-0.25985026359558105},{0.28841960430145264}}, "conv_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv#10");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.686335599515587e-05},{-inf},{inf}}, "conv_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*64*128);
    auto weights1 = om.constantInt(weightsData1,{3,3,64,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.003531770082190633},{-0.4776207208633423},{0.4229806363582611}}, "conv_1_weights#4");
    auto conv1 = om.conv(bias_c0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv_1#11");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.3850078175892122e-05},{-inf},{inf}}, "conv_1_bias#5");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.003921568859368563},{0.0},{1.0}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*128*128);
    auto weights2 = om.constantInt(weightsData2,{3,3,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.0034828942734748125},{-0.4314917325973511},{0.45664629340171814}}, "output_weights#7");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "output#12");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.365840853395639e-05},{-inf},{inf}}, "output_bias#8");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.003921568859368563},{0.0},{1.0}});

    om.output(bias_c2);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng keembay_adapt_model.dot -o keembay_adapt_model.png");
    system("dot -Tpng dma_model.dot -o dma_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
    system("dot -Tpng TransitiveReduction.dot -o TransitiveReduction.png");
    system("dot -Tpng deallocation_model_data.dot -o deallocation_model_data.png");
    system("dot -Tpng DmaControlFlows_model.dot -o DmaControlFlows_model.png");
    system("dot -Tpng InputOutputControlFlows_model.dot -o InputOutputControlFlows_model.png");
    system("flatc -t ../../schema/graphfile/src/schema/graphfile.fbs -- blob.bin");
}