#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST(MaxTopologicalCut, lessThanCMXMemory)
{

    mv::CompilationUnit unit("testMaxTopologicalCut");
    mv::OpModel& om = unit.model();
    
    auto input = om.input({112, 224, 3}, mv::DType("Float8"), mv::Order("CHW"));
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

    /*The max topological cut of the equivalent network in the PoC compiler is 491200*/
    ASSERT_EQ(maxTopologicalCutValue, 491200);

}
