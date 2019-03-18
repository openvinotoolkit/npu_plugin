#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(graph_coloring, case1)
{
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({112, 224, 3}, mv::DType("UInt8"), mv::Order("CHW"));
    std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(7*7*3*64);
    auto weights = om.constantInt(weightsData, {7, 7, 3, 64}, mv::DType("UInt8"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {2, 2}, {3, 3, 3, 3});

    om.output(conv);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);


    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    system("dot -Tpng original_model.dot -o original_model.png");
    system("dot -Tpng adapt_model.dot -o adapt_model.png");
    system("dot -Tpng final_model.dot -o final_model.png");
}