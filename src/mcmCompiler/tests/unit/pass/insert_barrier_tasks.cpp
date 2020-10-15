#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(insert_barrier_tasks, serial_path)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}); // one barrier

    om.output(conv1); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
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

    auto add1 = om.add({conv2, conv1});   // one barrier, #2

    om.output(add1); // one barrier for DMA out from CMX to DDR, #3

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.compilationDescriptor().remove("kmb_adapt", "GenerateSparsityMaps");
    unit.compilationDescriptor().remove("kmb_adapt", "GenerateWeightsTables");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o parallel_paths_final_model.png");

    std::vector<mv::Control::OpListIterator> barrierOps;
    for (auto opIt = cm.opBegin(); opIt != cm.opEnd(); ++opIt)
    {
        if (opIt->getOpType().find("BarrierTask") != std::string::npos)
            barrierOps.push_back(opIt);
    }

    int numChecks = 0;
    size_t expected_num_barriers = 4; // (3 + 1 barrier for prefetch)
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // barrier 0 is used by 2 convs (multiple consumers)
    for (auto b : barrierOps)
    {
        bool found = false;
        auto inputTensors = b->getInputTensor();
        for (auto childOp = b.leftmostChild(); childOp != cm.opEnd(); ++childOp)
        {
            if (childOp->getOpType() == "DPUTask"
                && childOp->getName().find("DPU_Conv_0") != std::string::npos)
            {
                found = true;
            }
        }

        if (found)
        {
            // check that we got the right parents (there should be 2)
            std::vector<std::string> childNames;
            for (auto childOp = b.leftmostChild(); childOp != cm.opEnd(); ++childOp)
            {
                if (childOp->getOpType() == "DPUTask")
                    childNames.push_back(childOp->getName());

            }
            EXPECT_TRUE(std::any_of(childNames.begin(),
                            childNames.end(),
                            [](std::string& s){ return s.find("DPU_Conv_1") != std::string::npos; } ));

            numChecks=numChecks+1;
            break;
        }
    }
    EXPECT_EQ(2, numChecks);   // coverage check
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

    auto add1 = om.add({conv2, conv1});   // one barrier, #3

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
    unit.compilationDescriptor().remove("kmb_adapt");
    unit.compilationDescriptor().remove("dma");
    unit.compilationDescriptor().remove("control_flows");
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o single_control_edge_final_model.png");

    std::vector<mv::Control::OpListIterator> barrierOps;
    for (auto opIt = cm.opBegin(); opIt != cm.opEnd(); ++opIt)
    {
        if (opIt->getOpType().find("BarrierTask") != std::string::npos)
            barrierOps.push_back(opIt);
    }

    int numChecks = 0;
    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // Check new barrier required by partial serialization
    for (auto b : barrierOps)
    {
        bool found = false;
        for (auto parentOp = b.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
        {
            if (parentOp->getOpType() == "DPUTask"
                && parentOp->getName().find("DPU_Conv_0") != std::string::npos)
            {
                found = true;
            }
        }

        if (found)
        {
            // Check that we got the right child
            std::vector<std::string> childDMATaskNames;
            for (auto childOp = b.leftmostChild(); childOp != cm.opEnd(); ++childOp)
            {
                if (childOp->getOpType() == "DMATask")
                    childDMATaskNames.push_back(childOp->getName());

            }
            EXPECT_TRUE(std::any_of(childDMATaskNames.begin(),
                            childDMATaskNames.end(),
                            [](const std::string& s){ return s.find("DMAAlignedConstant_1") != std::string::npos; } ));


            numChecks=numChecks+1;
            break;
        }
    }
    EXPECT_EQ(2, numChecks);   // coverage check
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

    auto add1 = om.add({conv2, conv1});   // one barrier, #3

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
    unit.compilationDescriptor().remove("kmb_adapt");
    unit.compilationDescriptor().remove("dma");
    unit.compilationDescriptor().remove("control_flows");
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");

    // number of workloads is not important -- the number of tasks that are tracked
    // by a barrier is important -- so remove workloads generation, since it
    // changes based on the algorithm chosen & can break this test. Nope...
    // this is failing as well. :`-(
    //unit.compilationDescriptor().remove("finalize","GenerateWorkloads");

    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o multiple_control_edges_final_model.png");

    std::vector<mv::Control::OpListIterator> barrierOps;
    for (auto opIt = cm.opBegin(); opIt != cm.opEnd(); ++opIt)
    {
        if (opIt->getOpType().find("BarrierTask") != std::string::npos)
            barrierOps.push_back(opIt);
    }

    int numChecks = 0;
    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // barriers affected by partial serialization should have extra producers
    for (auto b : barrierOps)
    {
        bool found = false;
        for (auto parentOp = b.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
        {
            if (parentOp->getOpType() == "DPUTask"
                && parentOp->getName().find("DPU_Conv_0") != std::string::npos)
            {
                found = true;
            }
        }

        if (found)
        {
            // check that we got the right parents (there should be 2)
            std::vector<std::string> parentNames;
            for (auto parentOp = b.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
            {
                if (parentOp->getOpType() == "DPUTask")
                    parentNames.push_back(parentOp->getName());

            }
            EXPECT_TRUE(std::any_of(parentNames.begin(),
                            parentNames.end(),
                            [](std::string& s){ return s.find("DPU_Conv_1") != std::string::npos; } ));

            // Check that we got the right child
            std::vector<std::string> childDMATaskNames;
            for (auto childOp = b.leftmostChild(); childOp != cm.opEnd(); ++childOp)
            {
                if (childOp->getOpType() == "DMATask")
                    childDMATaskNames.push_back(childOp->getName());

            }
            EXPECT_TRUE(std::any_of(childDMATaskNames.begin(),
                            childDMATaskNames.end(),
                            [](const std::string& s){ return s.find("DMAAlignedConstant_2") != std::string::npos; } ));


            numChecks=numChecks+2;
            break;
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

    auto add1 = om.add({conv2, conv1});   // one barrier, #3

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
    auto deAllocOp = om.getOp("DeallocateDMAAlignedConstant_0");
    auto holdOp = om.getOp("DMAAlignedConstant_1");
    cm.defineFlow(deAllocOp, holdOp);        // one barrier #5

    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.compilationDescriptor().remove("initialize");
    unit.compilationDescriptor().remove("adapt");
    unit.compilationDescriptor().remove("kmb_adapt");
    unit.compilationDescriptor().remove("dma");
    unit.compilationDescriptor().remove("control_flows");
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    // run passes after partial serilization, including insert barriers pass
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o dealloc_edge_final_model.png");
    std::vector<mv::Control::OpListIterator> barrierOps;
    for (auto opIt = cm.opBegin(); opIt != cm.opEnd(); ++opIt)
    {
        if (opIt->getOpType().find("BarrierTask") != std::string::npos)
            barrierOps.push_back(opIt);
    }

    int numChecks = 0;
    size_t expected_num_barriers = 6;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    // Check new barrier required by partial serialization
    for (auto b : barrierOps)
    {
        bool found = false;
        for (auto childOp = b.leftmostChild(); childOp != cm.opEnd(); ++childOp)
        {
            if (childOp->getOpType() == "DMATask"
                && childOp->getName().find("DMAAlignedConstant_1") != std::string::npos)
            {
                found = true;
            }
        }

        if (found)
        {
            // Check that we got the right parent. Dellocs are removed in the graph,
            // so the control flow would be from the preceding DPUTask (which, in this case)
            // is Conv_0. Verify that we have Conv_0 as one of the parents to this barrier.
            std::vector<std::string> parentNames;
            for (auto parentOp = b.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
            {
                if (parentOp->getOpType() == "DPUTask")
                    parentNames.push_back(parentOp->getName());

            }
            EXPECT_TRUE(std::any_of(parentNames.begin(),
                            parentNames.end(),
                            [](const std::string& s){ return s.find("Conv_0") != std::string::npos; } ));


            numChecks=numChecks+1;
            break;
        }
    }
    EXPECT_EQ(2, numChecks);   // coverage check
}

TEST(insert_barrier_tasks, static_index_assignment)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(om);

    auto input = om.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
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

    auto add0 = om.add({conv1, conv2});   // barrier

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
    unit.compilationDescriptor().remove("serialize");
    std::string optString = "Static";
    mv::Attribute option = optString;
    auto& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GlobalConfigParams", "barrier_index_assignment", option);

    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o static_barriers_final_model.png");

    auto sortedOps = cm.topologicalSort();

    std::vector<int> barrierOpIndices;
    for (auto op: sortedOps)
    {
        if (op->getOpType().find("BarrierTask") != std::string::npos)
            barrierOpIndices.push_back(op->get<mv::Barrier>("Barrier").getIndex());
    }

    size_t expected_num_barriers = 15;
    EXPECT_EQ(barrierOpIndices.size(), expected_num_barriers);

    // The barrier interference graph coloring algorithm (a.k.a. static index 
    // assignment algorithm) assigns indices between 0 and MAX_BARRIER_INDEX - 1.
    // Ensure that indices are in this range.
    const int MAX_BARRIER_INDEX = 8;
    EXPECT_TRUE(std::all_of(barrierOpIndices.begin(),
                barrierOpIndices.end(),
                [](int i){ return (i >= 0) && (i < MAX_BARRIER_INDEX - 1); }));

}

TEST(insert_barrier_tasks, dynamic_index_assignment)
{
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);

    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(unit.model());

    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    std::vector<double> weights3Data = mv::utils::generateSequence<double>(3*3*16*16);

    auto input = om.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});
    auto pool1 = om.maxPool(conv1, {2, 2}, {2, 2}, {0, 0, 0, 0});
    auto pool2 = om.maxPool(conv1, {4, 4}, {2, 2}, {1, 1, 1, 1});

    auto weights2 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv2 = om.conv(pool1, weights2, {1, 1}, {1, 1, 1, 1});

    auto weights3 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {1, 1, 1, 1});

    auto add1 = om.add({conv2, conv3});

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
    compDesc.setPassArg("GlobalConfigParams", "barrier_index_assignment", option);
    compDesc.setPassArg("AddWeightsDMATasks", "weights_prefetch", 2);

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    //unit.compilationDescriptor().remove("serialize");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o dynamic_barriers_final_model.png");

    auto sortedOps = cm.topologicalSort();

    std::vector<int> barrierOpIndices;
    for (auto op: sortedOps)
    {
        if (op->getOpType().find("BarrierTask") != std::string::npos)
            barrierOpIndices.push_back(op->get<mv::Barrier>("Barrier").getIndex());
    }

    size_t expected_num_barriers = 15;
    EXPECT_EQ(barrierOpIndices.size(), expected_num_barriers);
    EXPECT_TRUE(barrierOpIndices[0] == 0);

    for (size_t i = 1; i < barrierOpIndices.size(); ++i)
    {
        EXPECT_EQ(barrierOpIndices[i], barrierOpIndices[i-1] + 1);
    }
}


TEST(insert_barrier_tasks, weights_prefetch)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(unit.model());

    auto input = om.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});
    std::vector<double> weights3Data = mv::utils::generateSequence<double>(3*3*16*16);

    auto weights2 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv2 = om.conv(conv1, weights2, {1, 1}, {1, 1, 1, 1});

    auto weights3 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv3 = om.conv(conv2, weights3, {1, 1}, {1, 1, 1, 1});

    auto weights4 = om.constant(weights3Data, {3, 3, 16, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv4 = om.conv(conv3, weights4, {1, 1}, {1, 1, 1, 1});

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
    compDesc.setPassArg("GlobalConfigParams", "barrier_index_assignment", option);

    int dma_dependency = compDesc.getPassArg("dma","Singular","AddWeightsDMATasks","weights_prefetch");
    EXPECT_EQ(2, dma_dependency);     // default prefetch is 2
    int numChecks = 1;

    int prefetchTestVal = 3;
    compDesc.setPassArg("AddWeightsDMATasks", "weights_prefetch", prefetchTestVal);
    dma_dependency = compDesc.getPassArg("dma","Singular","AddWeightsDMATasks","weights_prefetch");
    EXPECT_EQ(prefetchTestVal, dma_dependency);
    numChecks++;

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().remove("serialize");
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng final_model.dot -o weights_prefetch_final_model.png");

    auto sortedOps = cm.topologicalSort();

    std::vector<int> barrierOpIndices;
    for (auto op: sortedOps)
    {
        if (op->getOpType().find("BarrierTask") != std::string::npos)
            barrierOpIndices.push_back(op->get<mv::Barrier>("Barrier").getIndex());
    }

    // With weights prefetch = 3 for this network, we expect 6 + 1 + (6 - (3+1)) barriers.
    // 6 ahead of the DPUTasks
    // 1 ahead of the DMATask
    // # total ops - (prefetch_val + 1) --> weights must be fetched after the prefetch_val-1 op
    // finishes executing.
    size_t expected_num_barriers = 9;
    EXPECT_EQ(barrierOpIndices.size(), expected_num_barriers);

}

static void RunTest(mv::CompilationUnit& unit, std::vector<int>& barrierOpIndices, const int reuseWindow)
{
    mv::OpModel& om = unit.model();
    mv::ControlModel cm(unit.model());

    auto input = om.input({56, 56, 16, 1}, mv::DType("Int8"), mv::Order("NCHW"));
    std::vector<int64_t> weightsData_1by1_16by64 = mv::utils::generateSequence<int64_t>(1*1*16*64);
    std::vector<int64_t> weightsData_1by1_64by64 = mv::utils::generateSequence<int64_t>(1*1*64*64);
    std::vector<int64_t> weightsData_3by3 = mv::utils::generateSequence<int64_t>(3*3*64*64);
    std::vector<int64_t> weightsData_3by3_2 = mv::utils::generateSequence<int64_t>(3*3*64*64);

    auto weights1 = om.constantInt(weightsData_1by1_16by64, {1, 1, 16, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});

    auto weights2 = om.constantInt(weightsData_3by3, {3, 3, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv2 = om.conv(conv1, weights2, {1, 1}, {1, 1, 1, 1});

    auto weights3 = om.constantInt(weightsData_1by1_64by64, {1, 1, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv3 = om.conv(conv2, weights3, {1, 1}, {1, 1, 1, 1});

    auto weights4 = om.constantInt(weightsData_3by3_2, {3, 3, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv4 = om.conv(conv3, weights4, {1, 1}, {1, 1, 1, 1});

    auto weights5 = om.constantInt(weightsData_1by1_64by64, {1, 1, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv5 = om.conv(conv4, weights5, {1, 1}, {1, 1, 1, 1});

    auto weights6 = om.constantInt(weightsData_3by3, {3, 3, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv6 = om.conv(conv5, weights6, {1, 1}, {1, 1, 1, 1});

    auto weights7 = om.constantInt(weightsData_1by1_64by64, {1, 1, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv7 = om.conv(conv6, weights7, {1, 1}, {1, 1, 1, 1});

    auto weights8 = om.constantInt(weightsData_3by3, {3, 3, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv8 = om.conv(conv7, weights8, {1, 1}, {1, 1, 1, 1});

    auto weights9 = om.constantInt(weightsData_1by1_64by64, {1, 1, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv9 = om.conv(conv8, weights9, {1, 1}, {1, 1, 1, 1});

    auto weights10 = om.constantInt(weightsData_1by1_64by64, {1, 1, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv10 = om.conv(conv9, weights10, {1, 1}, {1, 1, 1, 1});

    auto weights11 = om.constantInt(weightsData_1by1_64by64, {1, 1, 64, 64}, mv::DType("Int8"), mv::Order("NCWH"));
    auto conv11 = om.conv(conv10, weights11, {1, 1}, {1, 1, 1, 1});

    om.output(conv11); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);

    // This test is intended only for static barrier assignment
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "barrier_index_assignment", std::string("Static"));

    // Set the barrier reuse window size
    unit.compilationDescriptor().setPassArg("InsertBarrierTasks", "barrier_reuse_window", reuseWindow);

    // Effectively disable weights prefetch for this test by setting the prefetch number
    // to a large number
    unit.compilationDescriptor().setPassArg("AddWeightsDMATasks", "weights_prefetch", 200);

    unit.compilationDescriptor().remove("finalize","GenerateWorkloads");
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit.compilationDescriptor().remove("serialize","GenerateBlobKmb");

    // Clean up barrier indices after run (since we'll be creating and running multiple networks)
    unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    unit.initialize();
    unit.run();

    // For the network, with weights prefetch disabled, the barrier indices must go all the way to 7
    // before being reused, since we've set the reuse window to be 8. Indices would have otherwise
    // alternated between 0 & 1, since this is a pretty straightforward network.
    auto sortedOps = cm.topologicalSort();

    for (auto op: sortedOps)
    {
        if (op->getOpType().find("BarrierTask") != std::string::npos)
            barrierOpIndices.push_back(op->get<mv::Barrier>("Barrier").getIndex());
    }
}

TEST(insert_barrier_tasks, additional_interference)
{
    mv::CompilationUnit unit("testModel");
    mv::ControlModel cm(unit.model());

    std::vector<int> barrierOpIndices;
    int reuseWindow = 0;
    RunTest(unit, barrierOpIndices, reuseWindow);

    size_t expected_num_barriers = 12;
    EXPECT_EQ(barrierOpIndices.size(), expected_num_barriers);

    // The barrier interference graph coloring algorithm (a.k.a. static index
    // assignment algorithm) assigns indices between 0 and MAX_BARRIER_INDEX - 1.
    // Ensure that indices are in this range.
    EXPECT_TRUE(std::all_of(barrierOpIndices.begin(),
                            barrierOpIndices.end(),
                            [](int i){ return (i >= 0) && (i < 2); }));


    /////////////// Now run the same test with barrier reuse turned on. ////////////
    barrierOpIndices.clear();
    mv::CompilationUnit unit2("TestModel");
    reuseWindow = 8;
    RunTest(unit2, barrierOpIndices, reuseWindow);

    expected_num_barriers = 12;
    EXPECT_EQ(barrierOpIndices.size(), expected_num_barriers);

    EXPECT_TRUE(std::all_of(barrierOpIndices.begin(),
                            barrierOpIndices.end(),
                            [](int i){ return (i >= 0) && (i < 8); }));
}
