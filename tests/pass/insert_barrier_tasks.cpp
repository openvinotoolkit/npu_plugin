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

    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 2;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
}

/*
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

    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 4;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
}

TEST(insert_barrier_tasks, single_control_edge)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

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

    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");
    unit.compilationDescriptor().remove("finalize");
    unit.compilationDescriptor().remove("validate");
    unit.compilationDescriptor().remove("serialize");

    // run only the passes to build the task graph
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    // add edge to task graph, simulating partial serialization
    auto inbounddmaOp = om.getOp("DMAAlignedConstant_1");
    auto holdOp = om.getOp("DPU_MaxPool_1");
    cm.defineFlow(holdOp, inbounddmaOp);   // one barrier for inbound DMA
                                                                               unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.compilationDescriptor().remove("initialize");
    unit.compilationDescriptor().remove("adapt");
    unit.compilationDescriptor().remove("keembay_adapt");
    unit.compilationDescriptor().remove("dma");
    unit.compilationDescriptor().remove("control_flows");

    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 5;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
}

TEST(insert_barrier_tasks, multiple_control_edges)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({224, 224, 1}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*1*1);
    auto weights1 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weights2 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weights3 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weights4 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weights5 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {0, 0, 0, 0}); // barrier 1
    auto conv1a = om.conv(input, weights1, {1, 1}, {0, 0, 0, 0}); // barrier 2
    auto conv2 = om.conv(conv1, weights2, {1, 1}, {0, 0, 0, 0}); // barrier 3
    auto conv3 = om.conv(conv2, weights3, {1, 1}, {0, 0, 0, 0}); // barrier 4
    auto conv4 = om.conv(conv3, weights4, {1, 1}, {0, 0, 0, 0}); // barrier 5 
    auto pool1 = om.maxPool(conv4, {2, 2}, {1, 1}, {0, 0, 0, 0}); // barrier 6
    auto pool2 = om.maxPool(conv4, {4, 4}, {1, 1}, {1, 1, 1, 1}); // combined barrier with previous maxpool
    auto add1 = om.add(pool1, pool2);

    auto conv5 = om.conv(add1, weights5, {1, 1}, {1, 1, 1, 1}); // barrier 7

    om.output(conv5); // one barrier for DMA out from CMX to DDR: barrier 8

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");
    unit.compilationDescriptor().remove("finalize");
    unit.compilationDescriptor().remove("validate");
    unit.compilationDescriptor().remove("serialize");

    // run only the passes to build the task graph
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();


    // add edges to task graph, simulating partial serialization
    auto inbounddma4 = om.getOp("DMAAlignedConstant_3");
    auto conv0Op = om.getOp("DPU_Conv_0");
    auto conv1Op = om.getOp("DPU_Conv_1");
    auto holdOp = om.getOp("DMAAlignedConstant_2");
    cm.defineFlow(conv0Op, inbounddma4);   // barrier 9
    cm.defineFlow(conv1Op, inbounddma4);   // reuse barrier 9
    cm.defineFlow(holdOp, conv0Op);        // reuse barrier 1

    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.compilationDescriptor().remove("initialize");
    unit.compilationDescriptor().remove("adapt");
    unit.compilationDescriptor().remove("keembay_adapt");
    unit.compilationDescriptor().remove("dma");
    unit.compilationDescriptor().remove("control_flows");

    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();
    system("dot -Tpng final_model.dot -o final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 9;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);

    // barriers affected by partial serialization should have extra producers
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_0") EXPECT_EQ(3, b->get<mv::Barrier>("Barrier").getNumProducers());
        if (b->getName() == "BarrierTask_9") EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getNumProducers());
    }
}
*/

TEST(insert_barrier_tasks, static_index_assignment)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});
    auto pool1 = om.maxPool(conv1, {2, 2}, {2, 2}, {0, 0, 0, 0});
    auto pool2 = om.maxPool(conv1, {4, 4}, {2, 2}, {1, 1, 1, 1});

    std::vector<double> weights3Data = mv::utils::generateSequence<double>(3*3*16*16);
    auto weights2 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv2 = om.conv(pool1, weights2, {1, 1}, {1, 1, 1, 1});

    auto weights3 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {1, 1, 1, 1});

    auto add1 = om.add(conv2, conv3);

    auto weights4 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv4 = om.conv(add1, weights4, {1, 1}, {1, 1, 1, 1});

    auto weights5 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv5 = om.conv(conv4, weights5, {1, 1}, {1, 1, 1, 1});

    auto weights6 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv6 = om.conv(conv5, weights6, {1, 1}, {1, 1, 1, 1});

    om.output(conv6);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    std::string optString = "Static";
    mv::Attribute option = optString;
    auto& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("InsertBarrierTasks", "barrier_index_assignment", option);

    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");
    unit.compilationDescriptor().remove("serialize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o static_barriers_final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 9;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);

    // Expect the following due to graph coloring + static index assignment
    // barrier#0 -> index 0
    // barrier#6 -> index 2
    // barrier#9 -> index 1
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_0") EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_6") EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_9") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
    }
}

TEST(insert_barrier_tasks, dynamic_index_assignment)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});
    auto pool1 = om.maxPool(conv1, {2, 2}, {2, 2}, {0, 0, 0, 0});
    auto pool2 = om.maxPool(conv1, {4, 4}, {2, 2}, {1, 1, 1, 1});

    std::vector<double> weights3Data = mv::utils::generateSequence<double>(3*3*16*16);
    auto weights2 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv2 = om.conv(pool1, weights2, {1, 1}, {1, 1, 1, 1});

    auto weights3 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {1, 1, 1, 1});

    auto add1 = om.add(conv2, conv3);

    auto weights4 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv4 = om.conv(add1, weights4, {1, 1}, {1, 1, 1, 1});

    auto weights5 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv5 = om.conv(conv4, weights5, {1, 1}, {1, 1, 1, 1});

    auto weights6 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv6 = om.conv(conv5, weights6, {1, 1}, {1, 1, 1, 1});

    om.output(conv6);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    std::string optString = "Dynamic";
    mv::Attribute option = optString;
    auto& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("InsertBarrierTasks", "barrier_index_assignment", option);

    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");
    unit.compilationDescriptor().remove("serialize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o static_barriers_final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 9;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);

    // Expect the following due to dynamic index assignment
    // barrier#0 -> index 0
    // barrier#6 -> index 6
    // barrier#9 -> index 9
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_0") EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_6") EXPECT_EQ(6, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_9") EXPECT_EQ(9, b->get<mv::Barrier>("Barrier").getIndex());
    }

}
