#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "meta/include/mcm/op_model.hpp"

#include "iostream"
#include "fstream"

int main()
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16, 16, 15, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");

    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights1 = om.constant(weightsData1, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {0, 0, 0, 0});

    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    //compDesc.setPassArg("GenerateDot", "scope", std::string("ControlModel"));

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng keembay_adapt_model.dot -o keembay_adapt_model.png");
    system("dot -Tpng dma_model.dot -o dma_model.png");
    system("dot -Tpng control_model.dot -o control_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
}