#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(insert_barrier_tasks, serial_path)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}); // one barrier

    om.output(conv1); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().remove("finalize","TensorGraphColoring");
    unit.compilationDescriptor().remove("serialize");

    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
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

    auto input = om.input({64, 64, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
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
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

//    system("dot -Tpng final_model.dot -o parallel_paths_final_model.png");
    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 4;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // barrier 0 is used by 2 convs (multiple consumers)
    for (auto b : barrierOps)
    {
        //std::cout << " In parallel_paths test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumProducers() << std::endl;
        //std::cout << " In parallel_paths test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumConsumers() << std::endl;
        if (b->getName() == "BarrierTask_0") 
        {
            EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getNumProducers());
            EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getNumConsumers());
            numChecks=numChecks+2;
        }
    }
    EXPECT_EQ(3, numChecks);   // coverage check
}

TEST(insert_barrier_tasks, single_control_edge)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({64, 64, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
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
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();

//    system("dot -Tpng final_model.dot -o single_control_edge_final_model.png");
    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // Check new barrier required by partial serialization
    for (auto b : barrierOps)
    {
        //std::cout << " In single_control_edges test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumProducers() << std::endl;
        //std::cout << " In single_control_edges test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumConsumers() << std::endl;
        if (b->getName() == "BarrierTask_5")
        {
            EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getNumProducers());
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getNumConsumers());
            numChecks=numChecks+2;
        }
    }
    EXPECT_EQ(3, numChecks);   // coverage check
}

TEST(insert_barrier_tasks, multiple_control_edges)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({64, 64, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
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
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");

    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();

//    system("dot -Tpng final_model.dot -o multiple_control_edges_final_model.png");
    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // barriers affected by partial serialization should have extra producers
    for (auto b : barrierOps)
    {
        //std::cout << " In multiple_control_edges test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumProducers() << std::endl;
        //std::cout << " In multiple_control_edges test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumConsumers() << std::endl;
        if (b->getName() == "BarrierTask_5")
        {
            EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getNumProducers());
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getNumConsumers());
            numChecks=numChecks+2;
        }
    }
    EXPECT_EQ(3, numChecks);   // coverage check
}

TEST(insert_barrier_tasks, dealloc_edge)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({64, 64, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
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
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();

//    system("dot -Tpng final_model.dot -o dealloc_edge_final_model.png");
    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // Check new barrier required by partial serialization
    for (auto b : barrierOps)
    {
        //std::cout << " In static_index test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumProducers() << std::endl;
        //std::cout << " In static_index test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getNumConsumers() << std::endl;
        if (b->getName() == "BarrierTask_5")
        {
            EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getNumProducers());
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getNumConsumers());
            numChecks=numChecks+2;
        }
    }
    EXPECT_EQ(3, numChecks);   // coverage check
}

TEST(insert_barrier_tasks, static_index_assignment)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights0 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv0 = om.conv(input, weights0, {1, 1}, {1, 1, 1, 1});  // barrier
    auto pool0 = om.maxPool(conv0, {2, 2}, {2, 2}, {0, 0, 0, 0}); // barrier
    auto pool1 = om.maxPool(conv0, {4, 4}, {2, 2}, {1, 1, 1, 1}); // barrier
                                                                  // prefetch sparsity barrier

    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*16*16);
    auto weights1 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(pool0, weights1, {1, 1}, {1, 1, 1, 1});  // barrier

    auto weights2 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv2 = om.conv(pool1, weights2, {1, 1}, {1, 1, 1, 1});  // barrier
                                                                  // prefetch barrier

    auto add0 = om.add(conv1, conv2);   // barrier

    auto weights3 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(add0, weights3, {1, 1}, {1, 1, 1, 1});    // barrier
                                                                   // wts prefetch reuse barrier

    auto weights4 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv4 = om.conv(conv3, weights4, {1, 1}, {1, 1, 1, 1});   // barrier
                                                                   // wts prefetch barrier

    auto weights5 = om.constant(weights1Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv5 = om.conv(conv4, weights5, {1, 1}, {1, 1, 1, 1});   // barrier
                                                                   // wts prefetch barrier

    om.output(conv5);    // barrier (DMA)

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().remove("finalize","TensorGraphColoring");
    unit.compilationDescriptor().remove("serialize");
    std::string optString = "Static";
    mv::Attribute option = optString;
    auto& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("InsertBarrierTasks", "barrier_index_assignment", option);

    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

//    system("dot -Tpng final_model.dot -o static_barriers_final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 15;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // Expect reuse of barrier index numbers due to graph coloring + static index assignment
    for (auto b : barrierOps)
    {
//        std::cout << " In static_index test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getIndex() << std::endl;
        if (b->getName() == "BarrierTask_0")
        {
            EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_2")
        {   EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_6")
        {
            EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_8")
        {
            EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_1")
        {
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_4")
        {
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_7")
        {
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_9")
        {
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_3")
        {
            EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_5")
        {
            EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_10")
        {
            EXPECT_EQ(3, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_12")
        {
            EXPECT_EQ(3, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_14")
        {
            EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_16")
        {
            EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_18")
        {
            EXPECT_EQ(5, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
    }
    EXPECT_EQ(16, numChecks);   // coverage check
}

TEST(insert_barrier_tasks, dynamic_index_assignment)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
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

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().remove("finalize","TensorGraphColoring");
    unit.compilationDescriptor().remove("serialize");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

//    system("dot -Tpng final_model.dot -o dynamic_barriers_final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 15;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // Expect index assignment (no reuse) in dynamic mode
    for (auto b : barrierOps)
    {
//        std::cout << " In dynamic_index test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getIndex() << std::endl;
        if (b->getName() == "BarrierTask_0")
        {
            EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_1")
        {
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_2")
        {
            EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_3")
        {
            EXPECT_EQ(3, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_4")
        {
            EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_5")
        {
            EXPECT_EQ(5, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_6")
        {
            EXPECT_EQ(6, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_7")
        {
            EXPECT_EQ(7, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_8")
        {
            EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_9")
        {
            EXPECT_EQ(9, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_10")
        {
            EXPECT_EQ(10, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_12")
        {
            EXPECT_EQ(12, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_14")
        {
            EXPECT_EQ(14, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_16")
        {
            EXPECT_EQ(16, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_18")
        {
            EXPECT_EQ(18, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
    }
    EXPECT_EQ(16, numChecks);   // coverage check
}


TEST(insert_barrier_tasks, weights_prefetch)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
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

    int dma_dependency = compDesc.getPassArg("dma","Singular","AddDMATasks","weights_prefetch");
    EXPECT_EQ(2, dma_dependency);     // default prefetch is 2
    int numChecks = 1;
    compDesc.setPassArg("AddDMATasks", "weights_prefetch", 3);
    dma_dependency = compDesc.getPassArg("dma","Singular","AddDMATasks","weights_prefetch");
    EXPECT_EQ(3, dma_dependency);
    numChecks++;

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().remove("serialize");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

//    system("dot -Tpng final_model.dot -o weights_prefetch_final_model.png");

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 14;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // Expect index assignment (no reuse) in dynamic mode
    for (auto b : barrierOps)
    {
//        std::cout << " In weights_prefetch test: found " << b->getName() << " " << b->get<mv::Barrier>("Barrier").getIndex() << std::endl;
//        std::cout << "            numChecks = " << numChecks << std::endl;
        if (b->getName() == "BarrierTask_0")
        {
            EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_1")
        {
            EXPECT_EQ(1, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_2")
        {
            EXPECT_EQ(2, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_3")
        {
            EXPECT_EQ(3, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_4")
        {
            EXPECT_EQ(4, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_5")
        {
            EXPECT_EQ(5, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_6")
        {
            EXPECT_EQ(6, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_7")
        {
            EXPECT_EQ(7, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_8")
        {
            EXPECT_EQ(8, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_9")
        {
            EXPECT_EQ(9, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_10")
        {
            EXPECT_EQ(10, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_12")
        {
            EXPECT_EQ(12, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_14")
        {
            EXPECT_EQ(14, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
        if (b->getName() == "BarrierTask_16")
        {
            EXPECT_EQ(16, b->get<mv::Barrier>("Barrier").getIndex());
            numChecks++;
        }
    }
    EXPECT_EQ(17, numChecks);   // coverage check
}
