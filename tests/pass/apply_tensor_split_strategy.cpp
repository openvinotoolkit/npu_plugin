#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(apply_split_strategy, parse_strategy)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"), "myInput");
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"), "myWeights1");
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1, {{}, {}, {}, {}}, "myConv1"); // one barrier

    om.output(conv1); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.initialize();
    unit.run();
}