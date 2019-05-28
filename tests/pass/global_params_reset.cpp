#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

// The barrierCounter example shows the barrier Counter reset need and how to.
// BarrierCounter is a static variable that need to be reset between compilation
// unit runs, if the symbols (main program) is not reloaded. 3 compilation units
// are run below. 1st one doesn't reset the counter, so in the 2nd unit, the
// barrier counters don't start with 0. the 2nd unit resets the counter, so the
// 3rd unit sees the counters starting at 0.

TEST(global_params_reset, barrierCounter)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
    auto barrierOps = om.getOps("BarrierTask");
    auto input = om.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}); // one barrier
    om.output(conv1); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    //unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    unit.initialize();
    unit.run();
    // 2 barriers created
    int expected_num_barriers = 2;
    barrierOps = om.getOps("BarrierTask");
    std::sort(barrierOps.begin(), barrierOps.end(), [](mv::Data::OpListIterator a, mv::Data::OpListIterator b)
       {return (a->get<mv::Barrier>("Barrier").getID() < b->get<mv::Barrier>("Barrier").getID());});
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);

    for (auto b : barrierOps)
    {
        if (b->getName().find("BarrierTask_0") != std::string::npos)
        {
            EXPECT_EQ(0, b->get<mv::Barrier>("Barrier").getID());
        }
    }
    //----------------------------------------------------------------------------------------------//
    // 2nd compilation unit is created to show that the barrier IDs start with
    // nonzero index as the barrier Counter is not reset in the previous run
    mv::CompilationUnit unit2("testModel2");
    mv::OpModel& om2 = unit2.model();
    barrierOps = om2.getOps("BarrierTask");
    //below Assert is to check that the current status of barriers
    expected_num_barriers = 0; 
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    auto input_2 = om2.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto weights1_2 = om2.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1_2 = om2.conv(input_2, weights1_2, {1, 1}, {1, 1, 1, 1}); // one barrier
    om2.output(conv1_2); // one barrier for DMA out from CMX to DDR
    unit2.loadCompilationDescriptor(compDescPath);
    unit2.loadTargetDescriptor(mv::Target::ma2490);
    unit2.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit2.compilationDescriptor().remove("serialize");
    // now reset the barrier counter, to check on the 3rd compilation unit that the barrier count starts with 0.
    // *********below is a way to use the global params reset pass *************//
    unit2.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    unit2.initialize();
    unit2.run();
    barrierOps = om2.getOps("BarrierTask");
    std::sort(barrierOps.begin(), barrierOps.end(), [](mv::Data::OpListIterator a, mv::Data::OpListIterator b)
       {return (a->get<mv::Barrier>("Barrier").getID() < b->get<mv::Barrier>("Barrier").getID());});
    // expected 2 barriers
    expected_num_barriers = 2;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    // barriers start with ID=2 (2,3 are 2 barrier IDs in this unit). So below
    // comparison is to show IDs not equal to 0
    int barrierStartID = 0;
    //------------------------------------------------------------------------//

    mv::CompilationUnit unit3("testModel3");
    mv::OpModel& om3 = unit3.model();
    barrierOps = om3.getOps("BarrierTask");
    // Below Assert shows '0' starting barriers, ofcourse!
    expected_num_barriers = 0; 
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    auto input_3 = om3.input({28, 28, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto weights1_3 = om3.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1_3 = om3.conv(input_3, weights1_3, {1, 1}, {1, 1, 1, 1}); // one barrier
    om3.output(conv1_3); // one barrier for DMA out from CMX to DDR
    unit3.loadCompilationDescriptor(compDescPath);
    unit3.loadTargetDescriptor(mv::Target::ma2490);
    unit3.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    unit3.compilationDescriptor().remove("serialize");
    unit3.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    unit3.initialize();
    unit3.run();
    barrierOps = om3.getOps("BarrierTask");
    std::sort(barrierOps.begin(), barrierOps.end(), [](mv::Data::OpListIterator a, mv::Data::OpListIterator b)
       {return (a->get<mv::Barrier>("Barrier").getID() < b->get<mv::Barrier>("Barrier").getID());});
    expected_num_barriers = 2;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    barrierStartID = 0;
    // as barrier counter reset done in unit2, the barrier IDs in unit3 are 0,1
    for (auto b : barrierOps)
    {
        if (b->getName().find("BarrierTask_0") != std::string::npos)
        {
            EXPECT_EQ(barrierStartID, b->get<mv::Barrier>("Barrier").getID());
        }
    }
    
}
