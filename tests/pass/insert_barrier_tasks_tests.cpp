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
    mv::ControlModel cm(om);

    auto input = om.input({64, 64, 1}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*1*1);
    auto weight0 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight2 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv0 = om.conv(input, weight0, {1, 1}, {0, 0, 0, 0}); // one barrier, #0
    auto conv1 = om.conv(input, weight0, {1, 1}, {0, 0, 0, 0}); // REUSE barrier 0
    auto conv2 = om.conv(conv0, weight2, {1, 1}, {0, 0, 0, 0}); // one barrier, #1

    auto add1 = om.add(conv2, conv1);   // one barrier, #2

    om.output(add1); // one barrier for DMA out from CMX to DDR, #3

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("keembay_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("keembay_adapt", "GenerateWeightsTables");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o final_model.png");
    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 4;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);

    // barrier 0 is used by 2 convs (multiple consumers)
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_0") EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getNumProducers());
        if (b->getName() == "BarrierTask_0") EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getNumConsumers());
    }
}

TEST(insert_barrier_tasks, single_control_edge)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({64, 64, 1}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*1*1);
    auto weight0 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight1 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight2 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight3 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv0 = om.conv(input, weight0, {1, 1}, {0, 0, 0, 0}); // one barrier, #0
    auto conv1 = om.conv(input, weight1, {1, 1}, {0, 0, 0, 0}); // one barrier, #1
    auto conv2 = om.conv(conv0, weight2, {1, 1}, {0, 0, 0, 0}); // one barrier, #2

    auto add1 = om.add(conv2, conv1);   // one barrier, #3

    om.output(add1); // one barrier for DMA out from CMX to DDR, #4

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("finalize");
    unit.compilationDescriptor().remove("validate");
    unit.compilationDescriptor().remove("serialize");

    // run only the passes to build the task graph
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    // add an edge to task graph, simulating partial serialization
    auto inbounddmaOp = om.getOp("DMAAlignedConstant_1");
    auto conv0Op = om.getOp("DPU_Conv_0");
    cm.defineFlow(conv0Op, inbounddmaOp);   // one barrier for inbound DMA, from PS, #5

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

    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);

    // Check new barrier required by partial serialization
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getNumProducers());
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getNumConsumers());
    }
}

TEST(insert_barrier_tasks, multiple_control_edges)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({64, 64, 1}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*1*1);
    auto weight0 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight1 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight2 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv0 = om.conv(input, weight0, {1, 1}, {0, 0, 0, 0}); // one barrier, #0
    auto conv1 = om.conv(input, weight1, {1, 1}, {0, 0, 0, 0}); // one barrier, #1
    auto conv2 = om.conv(conv0, weight2, {1, 1}, {0, 0, 0, 0}); // one barrier, #2

    auto add1 = om.add(conv2, conv1);   // one barrier, #3

    om.output(add1); // one barrier for DMA out from CMX to DDR, #4

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("finalize");
    unit.compilationDescriptor().remove("validate");
    unit.compilationDescriptor().remove("serialize");

    // run only the passes to build the task graph
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    // add an edge to task graph, simulating partial serialization
    auto inbounddmaOp = om.getOp("DMAAlignedConstant_2");
    auto conv0Op = om.getOp("DPU_Conv_0");
    auto conv1Op = om.getOp("DPU_Conv_1");
    cm.defineFlow(conv0Op, inbounddmaOp);   // one barrier for inbound DMA, from PS, #5
    cm.defineFlow(conv1Op, inbounddmaOp);   // add dependencies (producers) to barrier #5

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

    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);

    // barriers affected by partial serialization should have extra producers
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getNumProducers());
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getNumConsumers());
    }
}

TEST(insert_barrier_tasks, dealloc_edge)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({64, 64, 1}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*1*1);
    auto weight0 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight1 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight2 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto weight3 = om.constant(weightsData, {1, 1, 1, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv0 = om.conv(input, weight0, {1, 1}, {0, 0, 0, 0}); // one barrier, #0
    auto conv1 = om.conv(input, weight1, {1, 1}, {0, 0, 0, 0}); // one barrier, #1
    auto conv2 = om.conv(conv0, weight2, {1, 1}, {0, 0, 0, 0}); // one barrier, #2

    auto add1 = om.add(conv2, conv1);   // one barrier, #3

    om.output(add1); // one barrier for DMA out from CMX to DDR, #4

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("finalize");
    unit.compilationDescriptor().remove("validate");
    unit.compilationDescriptor().remove("serialize");

    // run only the passes to build the task graph
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    // add an edge to task graph, simulating partial serialization
    auto holdOp = om.getOp("DMAAlignedConstant_1");
    auto deAllocOp = om.getOp("DeallocateDMAAlignedConstant_0");
    cm.defineFlow(deAllocOp, holdOp);        // one barrier #5

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

    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);

    // Check new barrier required by partial serialization
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getNumProducers());
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getNumConsumers());
    }
}

TEST(insert_barrier_tasks, static_index_assignment)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights0 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv0 = om.conv(input, weights0, {1, 1}, {1, 1, 1, 1});  // barrier #0 index 0
    auto pool0 = om.maxPool(conv0, {2, 2}, {2, 2}, {0, 0, 0, 0}); // barrier #3 index 2
    auto pool1 = om.maxPool(conv0, {4, 4}, {2, 2}, {1, 1, 1, 1}); // barrier #1 index 1

    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*16*16);
    auto weights1 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(pool0, weights1, {1, 1}, {1, 1, 1, 1});  // barrier #4 index 3

    auto weights2 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv2 = om.conv(pool1, weights2, {1, 1}, {1, 1, 1, 1});  // barrier #2 index 0

    auto add0 = om.add(conv1, conv2);   // barrier #5  index

    auto weights3 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(add0, weights3, {1, 1}, {1, 1, 1, 1});    // barrier #6  index 0
                                                                   // wts prefetch barrier #11

    auto weights4 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv4 = om.conv(conv3, weights4, {1, 1}, {1, 1, 1, 1});   // barrier #7  index 1
                                                                   // wts prefetch barrier #15

    auto weights5 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv5 = om.conv(conv4, weights5, {1, 1}, {1, 1, 1, 1});   // barrier #8  index 0
                                                                   // wts prefetch reuse barrier #11

    om.output(conv5);    // barrier #9  index 1

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    std::string optString = "Static";
    mv::Attribute option = optString;
    auto& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("InsertBarrierTasks", "barrier_index_assignment", option);

    unit.compilationDescriptor().remove("serialize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o static_barriers_final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 13;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);

    // Expect reuse of barrier index numbers due to graph coloring + static index assignment
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_0") EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_2") EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_6") EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_8") EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_1") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_7") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
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

    unit.compilationDescriptor().remove("serialize");

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o dynamic_barriers_final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 13;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);

    // Expect index assignment (no reuse) in dynamic mode
    for (auto b : barrierOps)
    {
        if (b->getName() == "BarrierTask_0") EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_1") EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_2") EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_3") EXPECT_EQ(3, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_4") EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_5") EXPECT_EQ(5, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_6") EXPECT_EQ(6, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_7") EXPECT_EQ(7, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_8") EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_9") EXPECT_EQ(9, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_10") EXPECT_EQ(10, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_11") EXPECT_EQ(11, b->get<mv::Barrier>("Barrier").getIndex());
        if (b->getName() == "BarrierTask_15") EXPECT_EQ(15, b->get<mv::Barrier>("Barrier").getIndex());
    }

}
