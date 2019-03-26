#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST(GlobalConfigParams, case_read)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({224, 224, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3*3*3*16);
    auto weights1 = om.constant(weightsData, {3, 3, 3, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(input, weights1, {1, 1}, {1, 1, 1, 1});
    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    //
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    //compDesc.a
    //compDesc.clear();
    std::string testString = "testing";
    compDesc.setPassArg("GlobalConfigParams", "test_string", testString);
    compDesc.setPassArg("GlobalConfigParams", "test_bool", true);
    compDesc.setPassArg("GlobalConfigParams", "test_float", 1.4);
    compDesc.setPassArg("GlobalConfigParams", "test_int", 2);

    //mv::Element dummyPassDesc("");
    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;
    mv::pass::PassRegistry::instance().find("GlobalConfigParams")->run(om, dummyTargDesc, compDesc, compOutput);
    
    // Check general model properties
    mv::Element returnedParams = om.getGlobalConfigParams();
    ASSERT_EQ(returnedParams.get<std::string>("test_string"), testString);
    ASSERT_EQ(returnedParams.get<bool>("test_bool"), true);
    ASSERT_EQ(returnedParams.get<float>("test_float"), 1.4);
    ASSERT_EQ(returnedParams.get<int>("test_float"), 2);
}
