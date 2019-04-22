#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(global_params_reset, barrierCounter)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();
//    mv::Barrier barriertest;

    auto barrierOps = om.getOps("BarrierTask");

    size_t expected_num_barriers = 0; //no barrier ops yet created
    size_t expected_barrier_ID = 0;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1}); // one barrier

    om.output(conv1); // one barrier for DMA out from CMX to DDR

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
    // reset the barrier counter to 0
    //unit.compilationDescriptor().addToGroup("root","GlobalParamsReset","Singular", false);
    unit.initialize();
    unit.run();


    expected_num_barriers = 2;
    barrierOps = om.getOps("BarrierTask");
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    int barrierID = 0;
    for (auto b : barrierOps)
    {
            EXPECT_EQ(barrierID, b->get<mv::Barrier>("Barrier").getID());
            std::cout<<"value of barriers: "<< b->get<mv::Barrier>("Barrier").getID()<<std::endl;
            barrierID++;
    }
// now donot reset the barrier counter.
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.compilationDescriptor().remove("initialize");
    unit.compilationDescriptor().remove("adapt");
    unit.compilationDescriptor().remove("keembay_adapt");
    unit.compilationDescriptor().remove("finalize");
    unit.compilationDescriptor().addGroup("finalize");
    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//  now donot reset the barrier counter. Below is commented out
    unit.initialize();
    unit.run();

    barrierOps = om.getOps("BarrierTask");

    expected_num_barriers = 2;
    ASSERT_EQ(barrierOps.size(), expected_num_barriers);
    barrierID = 0;
    for (auto b : barrierOps)
    {
            EXPECT_EQ(barrierID, b->get<mv::Barrier>("Barrier").getID());
            std::cout<<" 2 value of barriers: "<< b->get<mv::Barrier>("Barrier").getID()<<std::endl;
            barrierID++;
    }


}