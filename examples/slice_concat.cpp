#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>

#define CONFIG_SLICE_BRANCHES 2

#define CONFIG_INPUT_WIDTH 64
#define CONFIG_INPUT_HEIGHT 64
#define CONFIG_INPUT_CHANNELS 32

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

    auto input = om.input({CONFIG_INPUT_WIDTH,CONFIG_INPUT_HEIGHT,CONFIG_INPUT_CHANNELS,1},
                            mv::DType("UInt8"),
                            mv::Order::getZMajorID(4),
                            emptyQuantParams,
                            "input");

    std::vector<mv::Data::TensorIterator> slices0(CONFIG_SLICE_BRANCHES);
    std::vector<mv::Data::TensorIterator> slices1(CONFIG_SLICE_BRANCHES);

    size_t newHeight = CONFIG_INPUT_HEIGHT / CONFIG_SLICE_BRANCHES;
    size_t newWidth = CONFIG_INPUT_WIDTH / CONFIG_SLICE_BRANCHES;
    size_t startCoord = 0;

    for( int branch = 0; branch < CONFIG_SLICE_BRANCHES ; branch++)
    {
        auto slice = om.slice(input,
                                {0,startCoord,0,0},
                                {CONFIG_INPUT_WIDTH,newHeight,CONFIG_INPUT_CHANNELS,1},
                                emptyQuantParams,
                                "slice0_" + std::to_string(branch) + "_cmx_");
        slices0[branch] = slice;
        startCoord += newHeight;
    }


    auto concat0 = om.concat(slices0,"H",emptyQuantParams,"concat0_ddr_");

    startCoord = 0;
    for( int branch = 0; branch < CONFIG_SLICE_BRANCHES ; branch++)
    {
        auto slice = om.slice(concat0,
                                {startCoord,0,0,0},
                                {newWidth,CONFIG_INPUT_HEIGHT,CONFIG_INPUT_CHANNELS,1},
                                emptyQuantParams,
                                "slice1_" + std::to_string(branch) + "_cmx_");
        slices1[branch] = slice;
        startCoord += newWidth;
    }

    auto concat = om.concat(slices1,"W",emptyQuantParams,"concat1_ddr_");
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
