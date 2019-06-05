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
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    const mv::QuantizationParams& emptyQuantParams= {{}, {}, {}, {}};

    auto input = om.input({20,20,16,1},
                            mv::DType("UInt8"),
                            mv::Order::getZMajorID(4),
                            emptyQuantParams,
                            "input");

    auto slice0 = om.slice(input,
                            {0,0,0,0},
                            {10,20,16,1},
                            emptyQuantParams,
                            "slice0");
    auto slice1 = om.slice(input,
                            {10,0,0,0},
                            {10,20,16,1},
                            emptyQuantParams,
                            "slice1");

    std::vector<mv::Data::TensorIterator> slices(2);

    slices[0] = slice0;
    slices[1] = slice1;
    auto concat = om.concat(slices,"W",emptyQuantParams,"concat");
    auto output = om.output(concat);

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
}
