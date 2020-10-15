#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(streaming_pass, op_fission_conv0)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NHWC"));

    std::vector<int64_t> weights4Fixed = mv::utils::generateSequence<int64_t>((4*3*3*16), 0 , 1 );
    auto weights4 = om.constantInt(weights4Fixed, {4, 3, 3, 16}, mv::DType("UInt8"), mv::Order("NHWC"));
    auto conv4 = om.conv(input, weights4, {1, 1}, {0, 0, 0, 0});

    om.output(conv4);
    std::cout << "in fission test " << std::endl ;

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    int wtPrefetchTestVal = 11;
    auto& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("AddWeightsDMATasks", "weights_prefetch", wtPrefetchTestVal);

    unit.loadTargetDescriptor(mv::Target::ma2490);

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//    unit.compilationDescriptor().remove("serialize");
    
    unit.initialize();
    
    std::cout << "in fission test  running compile" << std::endl ;
    unit.run();
   
    std::cout << "in fission test DONE running compile" << std::endl ;
    system("");

    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 5;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    EXPECT_EQ(1, numChecks);   // coverage check
}

TEST(streaming_pass, op_fission_multiple_outputs)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NHWC"));

    std::vector<int64_t> weights4Fixed = mv::utils::generateSequence<int64_t>((4*3*3*16), 0 , 1 );
    auto weights4 = om.constantInt(weights4Fixed, {4, 3, 3, 16}, mv::DType("UInt8"), mv::Order("NHWC"));

    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>((1*1*1*16), 0 , 1 );
    auto weight0 = om.constantInt(weightsData, {1,1,16,1}, mv::DType("UInt8"), mv::Order("NHWC"));
    auto weight2 = om.constantInt(weightsData, {1,1,16,1}, mv::DType("UInt8"), mv::Order("NHWC"));
    auto conv0 = om.conv(input, weights4, {1, 1}, {0, 0, 0, 0});
    auto conv1 = om.conv(conv0, weight0, {1, 1}, {0, 0, 0, 0});
    auto conv2 = om.conv(conv0, weight2, {1, 1}, {0, 0, 0, 0});

    auto add1 = om.add({conv1, conv2});   // one barrier, #2

    om.output(add1); // one barrier for DMA out from CMX to DDR, #3

    std::cout << "in fission test diamond" << std::endl ;

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    int wtPrefetchTestVal = 11;
    auto& compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("AddWeightsDMATasks", "weights_prefetch", wtPrefetchTestVal);

    unit.loadTargetDescriptor(mv::Target::ma2490);

    unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//    unit.compilationDescriptor().remove("serialize");
    
    unit.initialize();
    
    std::cout << "in fission test  running compile" << std::endl ;
    unit.run();
   
    std::cout << "in fission test DONE running compile" << std::endl ;
    system("");

    auto barrierOps = om.getOps("BarrierTask");

    int numChecks = 0;
    size_t expected_num_barriers = 8;
    EXPECT_EQ(barrierOps.size(), expected_num_barriers);
    numChecks++;

    EXPECT_EQ(1, numChecks);   // coverage check
}
