#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

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
//    mv::Logger::setVerboseLevel(mv::VerboseLevel::Warning);
//    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    const mv::QuantizationParams& emptyQuantParams= {{}, {}, {}, {}};

    auto input = om.input({32,16,16,1},
                            mv::DType("UInt8"),
                            mv::Order::getZMajorID(4),
                            emptyQuantParams,
                            "input");

    auto slice0 = om.slice(input,
                            {0,0,0,0},
                            {16,16,16,1},
                            emptyQuantParams,
                            "slice0_cmx_");
    auto slice1 = om.slice(input,
                            {16,0,0,0},
                            {16,16,16,1},
                            emptyQuantParams,
                            "slice1_cmx_");
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(16*16);
    auto weights = om.constantInt(weightsData,{1,1,16,16},mv::DType("UInt8"),mv::Order::getZMajorID(4),emptyQuantParams,"weights");
    auto conv0 = om.conv(slice0,weights,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv0_cmx_");
    auto conv1 = om.conv(slice1,weights,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv1_cmx_");

    std::vector<mv::Data::TensorIterator> convs(2);

    convs[0] = conv0;
    convs[1] = conv1;
    auto concat = om.concat(convs,"W",emptyQuantParams,"concat_ddr_");
    auto output = om.output(concat);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490_streaming.json";
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
}
