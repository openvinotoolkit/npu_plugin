#include <algorithm>
#include "gtest/gtest.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/compiler/compilation_descriptor.hpp"

static void setPassReg()
{

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)> foo =
    [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& desc, mv::Element&, mv::Element&)
    {   
        if (desc.getTarget() == mv::Target::Unknown)
            throw mv::ArgumentError(model, "target", "unknown", "Test pass does not accept target decriptor"
                " with undefined target");
        mv::OpModel om(model);
        //om.clear();
        om.input({1}, mv::DType("Float16"), mv::Order("W"));
        om.addAttr(om.getInput(), "test", (bool)true);
    };

    mv::pass::PassRegistry::instance().enter("__TEST_pass1")
    .setDescription("Test pass entry")
    .setFunc(foo);

}

static void resetPassReg()
{
    mv::pass::PassRegistry::instance().remove("__TEST_pass1");
}


TEST(pass_registry, initialization)
{

    setPassReg();
    auto passList = mv::pass::PassRegistry::instance().list();
    ASSERT_TRUE(std::find(passList.begin(), passList.end(), "__TEST_pass1") != passList.end());
    resetPassReg();

}

TEST(pass_registry, run_pass)
{

    setPassReg();
    mv::OpModel model("testModel");
    
    mv::TargetDescriptor targetDesc;
    mv::CompilationDescriptor compDesc;
    mv::Element compOutput("CompilationOutput");
    ASSERT_THROW(mv::pass::PassRegistry::instance().run("__TEST_pass1", model, targetDesc, compDesc, compOutput), mv::ArgumentError);
    targetDesc.setTarget(mv::Target::ma2490);
    ASSERT_NO_THROW(mv::pass::PassRegistry::instance().run("__TEST_pass1", model, targetDesc, compDesc, compOutput));
    ASSERT_TRUE(model.getInput()->get<bool>("test"));
    resetPassReg();

}

