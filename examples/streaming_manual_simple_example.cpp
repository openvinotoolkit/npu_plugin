#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

//Based on conv_1 from migNetworkZoo/tools/kmb_test_generation/3_layer_small.csv
//Resnet 50, conv_1, Convolution, 56x56x64,  56x56x128, 3x3,1, UINT8, Yes,0.5
const int KERNEL_SIZE = 3;
const int ORIG_OUTPUT_CHANNELS = 128;
const int INPUT_CHANNELS = 64;
const int NUM_SPLITS = 2;
const int OUTPUT_CHANNELS = ORIG_OUTPUT_CHANNELS / NUM_SPLITS;
/*int main()
{
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({56, 56, INPUT_CHANNELS, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");

    std::vector<mv::Data::TensorIterator> convs(NUM_SPLITS);

    for (size_t i=0; i < NUM_SPLITS; i++)
    {
        std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(KERNEL_SIZE*KERNEL_SIZE*INPUT_CHANNELS*OUTPUT_CHANNELS);
        auto weights = om.constantInt(weightsData, {KERNEL_SIZE, KERNEL_SIZE, INPUT_CHANNELS, OUTPUT_CHANNELS}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "weightsss"+i);
        std::string name = "conv_"+std::to_string(i);
        convs[i] = om.conv(input, weights, {1, 1}, {1, 1, 1, 1}, 1, 1, {{},{},{},{}}, name);
    }
    if (NUM_SPLITS > 1)
    {
        auto concat = om.concat(convs);
        auto output = om.output(concat);
    }
    else
    {
        om.output(convs[0]);
    }

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
}*/
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
    auto input0 = om.input({56,56,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");

    std::vector<int64_t> weightsData0 = read_weights_from_file<int64_t>(path + "/projects/Fathom/src2/weights_bias/res2a_branch2a_weights#1.dat");
    std::vector<mv::Data::TensorIterator> weights(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> convs(NUM_SPLITS);
    std::vector<std::vector<int64_t>> weightsData(NUM_SPLITS);
    std::vector<std::vector<int64_t>> biasData(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> biases(NUM_SPLITS);
    std::vector<mv::Data::TensorIterator> biasesOp(NUM_SPLITS);

    std::vector<int64_t> biasWeightsData0 = read_weights_from_file<int64_t>(path + "/projects/Fathom/src2/weights_bias/res2a_branch2a_bias#2.dat");
    std::size_t biasSizePerSplit = biasWeightsData0.size()/NUM_SPLITS;
    std::size_t weightsSizePerSplit = weightsData0.size()/NUM_SPLITS;
    for (size_t i=0; i < NUM_SPLITS; i++)
    {
        weightsData[i].reserve(weightsSizePerSplit);
        auto itrBegin = weightsData0.begin() + i * weightsSizePerSplit;
        auto itrEnd = itrBegin + weightsSizePerSplit;
        std::copy(itrBegin, itrEnd, back_inserter(weightsData[i]));
        weights[i] = om.constantInt(weightsData[i],{1,1,64,64/NUM_SPLITS}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0028483974747359753},{-0.3647209107875824},{0.3616204559803009}}, "res2a_branch2a_weights#1" + std::to_string(i));
        //std::string name = "conv_1#4"+std::to_string(i);
        convs[i] = om.conv(input0, weights[i],  {1, 1}, {0, 0, 0, 0}, 1, 1, {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "res2a_branch2a#4" + std::to_string(i));

        biasData[i].reserve(biasSizePerSplit);
        itrBegin = biasWeightsData0.begin() + i * biasSizePerSplit;
        itrEnd = itrBegin + biasSizePerSplit;
        std::copy(itrBegin, itrEnd, back_inserter(biasData[i]));
        biases[i] = om.constantInt(biasData[i],{64/NUM_SPLITS}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{2.2340373106999323e-05},{-inf},{inf}}, "res2a_branch2a_bias#2" + std::to_string(i));
        biasesOp[i] = om.bias(convs[i], biases[i], {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});
    }
    //auto weights0 = om.constantInt(weightsData0,{3,3,64,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{127},{0.0033276514150202274},{-0.423709899187088},{0.4248411953449249}}, "conv_1_weights#1");
    //auto conv0 = om.conv(input0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.003921568859368563},{0.0},{1.0}}, "conv_1#4");

    if (NUM_SPLITS > 1)
    {
        auto concat = om.concat(biasesOp, "C", {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});
        auto output = om.output(concat);
    }
    else
    {
        om.output(biasesOp[0]);
    }
    //om.output(bias_c0);

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