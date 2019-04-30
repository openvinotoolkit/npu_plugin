#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

/*This test calculates max topological cut and does not perform partial serialisation as it is not required*/
TEST(MaxTopologicalCut, lessThanCMXMemory)
{

    mv::CompilationUnit unit("testMaxTopologicalCut");
    mv::OpModel& om = unit.model();

    auto input = om.input({112, 224, 3, 1}, mv::DType("Float8"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(7*7*3*64);
    auto weights = om.constant(weightsData, {7, 7, 3, 64}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {2, 2}, {3, 3, 3, 3});
    om.output(conv);
    
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    mv::ControlModel cm(om);

    auto output = cm.getOutput();
    int maxTopologicalCutValue;

    /*Get the max topological cut value*/
    if(output->hasAttr("MaxTopologicalCutValue"))
        maxTopologicalCutValue = output->get<int>("MaxTopologicalCutValue");

    /*The max topological cut of the equivalent network in the PoC compiler is 492032*/
    ASSERT_EQ(maxTopologicalCutValue, 492032);

}

/* This test perform max topological cut and inserts two partial serialisation edges*/
/* The edges inserted are non optimal at present and should be inserted to minimise an increase in critical path*/
/* It is not possible to do a direct comparison with the PoC compiler until the edges are inserted optimally*/
TEST(MaxTopologicalCut, greaterThanCMXMemory)
{

    mv::CompilationUnit unit("testMaxTopologicalCut");
    mv::OpModel& om = unit.model();

    auto input = om.input({40, 40, 3, 1}, mv::DType("Float8"), mv::Order("NCHW"));

    std::vector<double> weightsData = mv::utils::generateSequence<double>(7*7*3*90);
    auto weights = om.constant(weightsData, {7, 7, 3, 90}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {3, 3, 3, 3});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(7*7*90*90);
    auto weights1 = om.constant(weightsData1, {7, 7, 90, 90}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {3, 3, 3, 3});

    std::vector<double> weightsData2 = mv::utils::generateSequence<double>(7*7*90*90);
    auto weights2 = om.constant(weightsData2, {7, 7, 90, 90}, mv::DType("Float8"), mv::Order("NCWH"));
    auto conv2 = om.conv(conv1, weights2, {1, 1}, {3, 3, 3, 3});

    om.output(conv2);
    
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    mv::ControlModel cm(om);

    auto output = cm.getOutput();
    int maxTopologicalCutValue;
    
    /*Get the max topological cut value*/
    if(output->hasAttr("MaxTopologicalCutValue"))
        maxTopologicalCutValue = output->get<int>("MaxTopologicalCutValue");

    ASSERT_EQ(maxTopologicalCutValue, 688320);

}