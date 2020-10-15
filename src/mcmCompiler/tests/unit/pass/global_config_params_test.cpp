#include "gtest/gtest.h"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

TEST(GlobalConfigParams, case_read)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    mv::Element testPassDesc("GlobalConfigParams");
    testPassDesc.set("test_string", std::string("testing"));
    testPassDesc.set("test_bool", true);
    testPassDesc.set("test_double", 1.4);
    testPassDesc.set("test_int", 2);

    mv::TargetDescriptor dummyTargDesc;
    mv::Element compOutput("CompilationOutput");
    mv::pass::PassRegistry::instance().find("GlobalConfigParams")->run(om, dummyTargDesc, testPassDesc, compOutput);
    
    // Check global params
    std::shared_ptr<mv::Element> returnedParams = om.getGlobalConfigParams();
    std::string s = returnedParams->get<std::string>("test_string");
    
    ASSERT_EQ(returnedParams->get<std::string>("test_string"), std::string("testing"));
    ASSERT_EQ(returnedParams->get<bool>("test_bool"), true);
    ASSERT_EQ(returnedParams->get<double>("test_double"), 1.4);
    ASSERT_EQ(returnedParams->get<int>("test_int"), 2);
}

TEST(GlobalConfigParams, case_readlast)
{
    mv::CompilationUnit unit("testModel");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    mv::OpModel& om = unit.model();

    mv::Element testPassDesc("GlobalConfigParams");
    testPassDesc.set("test_string", std::string("testing"));
    testPassDesc.set("test_bool", true);
    testPassDesc.set("test_double", 1.4);
    testPassDesc.set("test_int", 2);

    mv::TargetDescriptor dummyTargDesc;
    dummyTargDesc.setTarget(mv::Target::ma2490);
    mv::Element compOutput("CompilationOutput");
    mv::pass::PassRegistry::instance().find("GlobalConfigParams")->run(om, dummyTargDesc, testPassDesc, compOutput);

    //run another pass to see if config is still correct after
    mv::Element testPassDescDot("GenerateWorkloads");
    testPassDescDot.set("costfunction", std::string("balanced"));
    mv::pass::PassRegistry::instance().find("GenerateWorkloads")->run(om, dummyTargDesc, testPassDescDot, compOutput);

    // Check global params still correct
    std::shared_ptr<mv::Element> returnedParams = om.getGlobalConfigParams();
    std::string s = returnedParams->get<std::string>("test_string");
    
    ASSERT_EQ(returnedParams->get<std::string>("test_string"), std::string("testing"));
    ASSERT_EQ(returnedParams->get<bool>("test_bool"), true);
    ASSERT_EQ(returnedParams->get<double>("test_double"), 1.4);
    ASSERT_EQ(returnedParams->get<int>("test_int"), 2);
}
