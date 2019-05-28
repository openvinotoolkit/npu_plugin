#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>
//recorded from res2a_branch2a, and then split
const int NUM_SPLITS = 2;
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
    auto input0 = om.input({56,56,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData0 = read_weights_from_file<int64_t>(path + "/examples/data/res2a_branch2a_weights#1.dat");
    std::vector<mv::Data::TensorIterator> weights(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> convs(NUM_SPLITS);
    std::vector<std::vector<int64_t>> weightsData(NUM_SPLITS);
    std::vector<std::vector<int64_t>> biasData(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> biases(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> biasesOp(NUM_SPLITS);

    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t>(path + "/examples/data//res2a_branch2a_bias#2.dat");
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
        weights[i] = om.constantInt(weightsData[i],{1,1,64,64/NUM_SPLITS}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.002591547090560198},{-0.3342282176017761},{0.3266163170337677}}, "res2a_branch2a_weights#1" + name_suffix);
        convs[i] = om.conv(input0, weights[i],  {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "res2a_branch2a#4" + name_suffix);

        biasData[i].reserve(biasSizePerSplit);
        itrBegin = biasWeightsData0.begin() + i * biasSizePerSplit;
        itrEnd = itrBegin + biasSizePerSplit;
        std::copy(itrBegin, itrEnd, back_inserter(biasData[i]));
        biases[i] = om.constantInt(biasData[i],{64/NUM_SPLITS}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.0325860532466322e-05},{-inf},{inf}}, "res2a_branch2a_bias#2weights" + name_suffix);
        biasesOp[i] = om.bias(convs[i], biases[i], {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}},  "res2a_branch2a_bias#2" + name_suffix);
    }

    if (NUM_SPLITS > 1)
    {
        auto concat = om.concat(biasesOp, "C", {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});
        auto output = om.output(concat);
    }
    else
    {
        om.output(biasesOp[0]);
    }

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