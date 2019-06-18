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

    auto input = om.input({32,32,128,1},
                            mv::DType("UInt8"),
                            mv::Order::getZMajorID(4),
                            emptyQuantParams,
                            "Input");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t>(128*64);
    auto weights0 = om.constantInt(weightsData0,
                                    {1,1,128,64},
                                    mv::DType("UInt8"),
                                    mv::Order::getZMajorID(4),
                                    emptyQuantParams,
                                    "weights0");

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t>(64*64);
    auto weights1 = om.constantInt(weightsData1,
                                    {1,1,64,64},
                                    mv::DType("UInt8"),
                                    mv::Order::getZMajorID(4),
                                    emptyQuantParams,
                                    "weights1");

    auto conv0 = om.conv(input,weights0,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv0_cmx_");
    auto conv1 = om.conv(conv0,weights1,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv1_cmx_");
    auto output = om.output(conv1);

//    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490_streaming.json";
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    auto& compDesc = unit.compilationDescriptor();

    std::string barrierStrategy = "Dynamic";
    compDesc.setPassArg("GlobalConfigParams", "barrier_index_assignment", barrierStrategy);
    auto ndpu = 1;
    compDesc.setPassArg("GlobalConfigParams", "Number_of_DPUs", ndpu);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    std::cout << "CONV_MULTI_PASS EXAMPLE: calling unit run" << std::endl ;
    unit.run();

}
