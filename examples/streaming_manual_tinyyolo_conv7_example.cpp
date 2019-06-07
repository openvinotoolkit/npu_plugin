#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>
//recorded from tinyyolo conv_7, and then split
//Kernel = 3x3x1024x1024 ~= 9M => split by 16 => ~0.56M
const int NUM_SPLITS = 16;
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
    std::string path = std::getenv("MCM_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({7,7,1024,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData0 = read_weights_from_file<int64_t>(path + "/examples/data/conv7_relu_weights#1.dat");
    std::vector<mv::Data::TensorIterator> weights(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> convs(NUM_SPLITS);
    std::vector<std::vector<int64_t>> weightsData(NUM_SPLITS);
    std::vector<std::vector<int64_t>> biasData(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> biases(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> biasesOp(NUM_SPLITS);

    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t>(path + "/examples/data/conv7_relu_bias#2.dat");
    std::size_t biasSizePerSplit = biasWeightsData0.size()/NUM_SPLITS;
    std::size_t weightsSizePerSplit = weightsData0.size()/NUM_SPLITS;
    std::string name_suffix = "";
    for (size_t i=0; i < NUM_SPLITS; i++)
    {
        weightsData[i].reserve(weightsSizePerSplit);
        auto itrBegin = weightsData0.begin() + i * weightsSizePerSplit;
        auto itrEnd = itrBegin + weightsSizePerSplit;
        std::copy(itrBegin, itrEnd, back_inserter(weightsData[i]));
        if (NUM_SPLITS > 1)
            name_suffix = std::to_string(i);
        weights[i] = om.constantInt(weightsData[i],{3,3,1024,1024/NUM_SPLITS}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{134},{0.0039022150449454784},{-0.5223554372787476},{0.47270941734313965}}, "conv7_relu_weights#1" + name_suffix);
        convs[i] = om.conv(input0, weights[i],   {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv7_relu#4" + name_suffix);

        biasData[i].reserve(biasSizePerSplit);
        itrBegin = biasWeightsData0.begin() + i * biasSizePerSplit;
        itrEnd = itrBegin + biasSizePerSplit;
        std::copy(itrBegin, itrEnd, back_inserter(biasData[i]));
        biases[i] = om.constantInt(biasData[i],{1024/NUM_SPLITS}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{3.0605609936174005e-05},{-inf},{inf}}, "conv7_relu_bias#2weights" + name_suffix);
        biasesOp[i] = om.bias(convs[i], biases[i],  {{0},{0.003921568859368563},{0.0},{1.0}},  "conv7_relu_bias#2" + name_suffix);
    }

    if (NUM_SPLITS > 1)
    {
        //TODO need to quantize this layer but since it's not a DPU task for now I am passing the shift/mult
        auto concat = om.concat(biasesOp, "C", {{0},{0.003921568859368563},{0.0},{1.0}, {22}, {32734}});
        auto output = om.output(concat);
    }
    else
    {
        om.output(biasesOp[0]);
    }

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    auto& compDesc = unit.compilationDescriptor();
    std::string barrierStrategy = "Dynamic";
    compDesc.setPassArg("GlobalConfigParams", "barrier_index_assignment", barrierStrategy);
    auto ndpu = 1;
    compDesc.setPassArg("GlobalConfigParams", "Number_of_DPUs", ndpu);

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
    //system("flatc -t ../../schema/graphfile/src/schema/graphfile.fbs -- output/vpu2.blob");
}