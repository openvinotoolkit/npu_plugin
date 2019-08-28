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
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    const mv::QuantizationParams& emptyQuantParams= {{}, {}, {}, {}};

    auto input = om.input({32,32,128,1},
                            mv::DType("UInt8"),
                            mv::Order::getZMajorID(4),
                            emptyQuantParams,
                            "input");

    auto slice0 = om.slice(input,
                                {0,0,0,0},
                                {16,32,128,1},
                                emptyQuantParams,
                                "slice0_level0_input_");

    auto slice1 = om.slice(input,
                                {16,0,0,0},
                                {16,32,128,1},
                                emptyQuantParams,
                                "slice1_level0_input_");

    auto slice0_0 = om.slice(slice0,
                                {0,0,0,0},
                                {16,16,128,1},
                                emptyQuantParams,
                                "slice0_level1_cmx_");

    auto slice0_1 = om.slice(slice0,
                                {0,16,0,0},
                                {16,16,128,1},
                                emptyQuantParams,
                                "slice1_level1_cmx_");

    auto slice1_0 = om.slice(slice1,
                                {0,0,0,0},
                                {16,16,128,1},
                                emptyQuantParams,
                                "slice2_level1_cmx_");
    auto slice1_1 = om.slice(slice1,
                                {0,16,0,0},
                                {16,16,128,1},
                                emptyQuantParams,
                                "slice3_level1_cmx_");

    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(128*64);
    auto weights = om.constantInt(weightsData,
                                    {1,1,128,64},
                                    mv::DType("UInt8"),
                                    mv::Order::getZMajorID(4),
                                    emptyQuantParams,
                                    "weights");

    auto conv0_0 = om.conv(slice0_0,weights,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv0_0_cmx_");
    auto conv0_1 = om.conv(slice0_1,weights,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv0_1_cmx_");
    auto conv1_0 = om.conv(slice1_0,weights,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv1_0_cmx_");
    auto conv1_1 = om.conv(slice1_1,weights,{1,1},{0,0,0,0},1,1,emptyQuantParams,"conv1_1_cmx_");

    std::vector<mv::Data::TensorIterator> convs0(2);
    std::vector<mv::Data::TensorIterator> convs1(2);

    convs0[0] = conv0_0;
    convs0[1] = conv0_1;
    convs1[0] = conv1_0;
    convs1[1] = conv1_1;

    auto concat0_level0 = om.concat(convs0,"H",emptyQuantParams,"concat0_level0_output_");
    auto concat1_level0 = om.concat(convs1,"H",emptyQuantParams,"concat1_level0_output_");

    std::vector<mv::Data::TensorIterator> concats(2);
    concats[0]=concat0_level0;
    concats[1]=concat1_level0;

    auto concat = om.concat(concats,"W",emptyQuantParams,"concat_level1_ddr_");

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
