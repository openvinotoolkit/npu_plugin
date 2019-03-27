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

    mv::Element testyPassDesc("GlobalConfigParams");
    testyPassDesc.set("test_string", std::string("testing"));
    testyPassDesc.set("test_bool", true);
    testyPassDesc.set("test_double", 1.4);
    testyPassDesc.set("test_int", 2);

    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;
    mv::pass::PassRegistry::instance().find("GlobalConfigParams")->run(om, dummyTargDesc, testyPassDesc, compOutput);
    
    // Check general model properties
    mv::Element returnedParams = om.getGlobalConfigParams();
    ASSERT_EQ(returnedParams.get<std::string>("test_string"), std::string("testing"));
    ASSERT_EQ(returnedParams.get<bool>("test_bool"), true);
    ASSERT_EQ(returnedParams.get<double>("test_double"), 1.4);
    ASSERT_EQ(returnedParams.get<int>("test_int"), 2);
}
