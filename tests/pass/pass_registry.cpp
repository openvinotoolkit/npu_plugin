#include "gtest/gtest.h"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"

std::function<void(mv::ComputationModel&, mv::TargetDescriptor&)> foo = [](mv::ComputationModel& model, mv::TargetDescriptor& desc)
{   
    if (desc.getTarget() == mv::Target::Unknown)
        throw mv::ArgumentError("target", "unknown", "Test pass does not accept target decriptor"
            " with undefined target");
    mv::OpModel om(model);
    om.clear();
    om.input(mv::Shape(1), mv::DType::Unknown, mv::Order::LastDimMajor, "customInput");
    om.addAttr(om.getInput(), "test", mv::Attribute(mv::AttrType::BoolType, true));

};

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(pass1)
        .setGenre(mv::PassGenre::Adaptation)
        .setDescription("Test pass entry")
        .setFunc(foo);

    }
}


TEST(pass_registry, initialization)
{

    mv::OpModel model;
    
    mv::TargetDescriptor desc;
    ASSERT_THROW(mv::pass::PassRegistry::instance().run("pass1", model, desc), mv::ArgumentError);
    desc.setTarget(mv::Target::ma2480);
    mv::pass::PassRegistry::instance().run("pass1", model, desc);
    ASSERT_TRUE(model.getInput()->getAttr("test").getContent<bool>());

}

