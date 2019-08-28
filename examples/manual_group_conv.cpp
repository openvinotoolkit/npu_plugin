#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

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

    auto input = om.input({128,128,64,1},
                            mv::DType("UInt8"),
                            mv::Order::getZMajorID(4),
                            emptyQuantParams,
                            "input");

    auto slice0 = om.slice(input,
                            {0,0,0,0},
                            {128,128,32,1},
                            emptyQuantParams,
                            "slice_grou0_input_");
    auto slice1 = om.slice(input,
                            {0,0,0,0},
                            {128,128,32,1},
                            emptyQuantParams,
                            "slice_group1_input_");

    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(32*32*3*3);

    auto weights0 = om.constantInt(weightsData,{3,3,32,32},mv::DType("UInt8"),mv::Order::getZMajorID(4),emptyQuantParams,"weights0");
    auto weights1 = om.constantInt(weightsData,{3,3,32,32},mv::DType("UInt8"),mv::Order::getZMajorID(4),emptyQuantParams,"weights1");

    auto conv0 = om.conv(slice0,weights0,{1,1},{1,1,1,1},1,1,emptyQuantParams,"conv0_ddr_");
    auto conv1 = om.conv(slice1,weights1,{1,1},{1,1,1,1},1,1,emptyQuantParams,"conv1_ddr_");

    std::vector<mv::Data::TensorIterator> convs(2);
    convs[0] = conv0;
    convs[1] = conv1;
    auto concat = om.concat(convs,"C",emptyQuantParams,"concat_ddr_");
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

}
