//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

template <typename T> std::vector<T> read_weights_from_file(std::string input_file)
{
    std::ifstream file;
    T inputString;
    std::vector<T> data;
    file.open(input_file);
    while(file>>inputString)
        data.push_back(inputString);
    file.close();
    return data;
}

int main()
{
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({32,32,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*16*1);
    auto weights0 = om.constantInt(weightsData0,{3,3,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.002290877280756831},{-0.28270426392555237},{0.30146944522857666}}, "test_weights#1");
    auto depthConv0 = om.depthwiseConv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "test#4");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (16);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.796766446204856e-05},{-inf},{inf}}, "test_bias#2");
    auto bias_cd0 = om.bias(depthConv0, biasWeights0, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_cd0);

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
