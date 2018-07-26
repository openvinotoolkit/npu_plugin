#include <algorithm>
#include "gtest/gtest.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"

static void setPassReg()
{

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> foo = 
    [](mv::ComputationModel& model, mv::TargetDescriptor& desc, mv::json::Object&, mv::json::Object&)
    {   
        if (desc.getTarget() == mv::Target::Unknown)
            throw mv::ArgumentError("target", "unknown", "Test pass does not accept target decriptor"
                " with undefined target");
        mv::OpModel om(model);
        om.clear();
        om.input(mv::Shape(1), mv::DType::Unknown, mv::Order::LastDimMajor, "customInput");
        om.addAttr(om.getInput(), "test", mv::Attribute(mv::AttrType::BoolType, true));

    };

    mv::pass::PassRegistry::instance().enter("__TEST_pass1")
    .setGenre(mv::PassGenre::Adaptation)
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
    mv::OpModel model;
    auto passList = mv::pass::PassRegistry::instance().list();
    ASSERT_TRUE(std::find(passList.begin(), passList.end(), "__TEST_pass1") != passList.end());
    resetPassReg();

}

TEST(pass_registry, run_pass)
{

    setPassReg();
    mv::OpModel model;
    
    mv::TargetDescriptor targetDesc;
    mv::json::Object compDesc;
    mv::json::Object compOutput;
    ASSERT_THROW(mv::pass::PassRegistry::instance().run("__TEST_pass1", model, targetDesc, compDesc, compOutput), mv::ArgumentError);
    targetDesc.setTarget(mv::Target::ma2480);
    mv::pass::PassRegistry::instance().run("__TEST_pass1", model, targetDesc, compDesc, compOutput);
    ASSERT_TRUE(model.getInput()->getAttr("test").getContent<bool>());
    resetPassReg();

}

