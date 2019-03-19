#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(insert_barrier_tasks, serial_path)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}); // one barrier

    om.output(conv1); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("adapt", "GenerateWeightsTables");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 2;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
}

TEST(insert_barrier_tasks, parallel_paths)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}); // one barrier
    auto pool1 = om.maxPool(conv1, {2, 2}, {2, 2}, {0, 0, 0, 0}); // one barrier
    auto pool2 = om.maxPool(conv1, {4, 4}, {2, 2}, {1, 1, 1, 1}); // combined barrier with previous maxpool

    auto add1 = om.add(pool1, pool2);

    std::vector<double> weights3Data = mv::utils::generateSequence<double>(3*3*16*32);
    auto weights3 = om.constant(weights3Data, {3, 3, 16, 32}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(add1, weights3, {1, 1}, {1, 1, 1, 1}); // one barrier

    om.output(conv3); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("adapt", "GenerateWeightsTables");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 4;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
}