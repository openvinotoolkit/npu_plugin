#include "gtest/gtest.h"
#include "include/mcm/pass/pass_manager.hpp"
#include "include/mcm/computation/model/op_model.hpp"

std::function<void(mv::ComputationModel&, mv::TargetDescriptor&)> adaptPass1 = [](mv::ComputationModel& model, mv::TargetDescriptor&)
{   
    mv::OpModel om(model);
    om.addAttr(om.getInput(), "adapt1", mv::Attribute(mv::AttrType::BoolType, true));

};

std::function<void(mv::ComputationModel&, mv::TargetDescriptor&)> adaptPass2 = [](mv::ComputationModel& model, mv::TargetDescriptor&)
{   
    mv::OpModel om(model);
    om.addAttr(om.getInput(), "adapt2", mv::Attribute(mv::AttrType::BoolType, true));

};

std::function<void(mv::ComputationModel&, mv::TargetDescriptor&)> validPass1 = [](mv::ComputationModel& model, mv::TargetDescriptor&)
{   
    mv::OpModel om(model);
    if (!om.getInput()->hasAttr("valid"))
        om.addAttr(om.getInput(), "valid", mv::Attribute(mv::AttrType::UnsingedType, 1U));
    else
    {
        auto attr = om.getInput()->getAttr("valid");
        attr.setContent<unsigned>(attr.getContent<unsigned>() + 1);
    }

};

std::function<void(mv::ComputationModel&, mv::TargetDescriptor&)> optPass1 = [](mv::ComputationModel& model, mv::TargetDescriptor&)
{   
    mv::OpModel om(model);
    om.addAttr(om.getInput(), "opt1", mv::Attribute(mv::AttrType::BoolType, true));

};

std::function<void(mv::ComputationModel&, mv::TargetDescriptor&)> finalPass1 = [](mv::ComputationModel& model, mv::TargetDescriptor&)
{   
    mv::OpModel om(model);
    om.addAttr(om.getInput(), "final1", mv::Attribute(mv::AttrType::BoolType, true));

};

std::function<void(mv::ComputationModel&, mv::TargetDescriptor&)> serialPass1 = [](mv::ComputationModel& model, mv::TargetDescriptor&)
{   
    mv::OpModel om(model);
    om.addAttr(om.getInput(), "serial1", mv::Attribute(mv::AttrType::BoolType, true));

};

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AdaptPass1)
        .setGenre(mv::PassGenre::Adaptation)
        .setDescription("Test execution of a scheduled adaptation pass")
        .setFunc(adaptPass1);

        MV_REGISTER_PASS(AdaptPass2)
        .setGenre(mv::PassGenre::Adaptation)
        .setDescription("Test execution of a scheduled adaptation pass")
        .setFunc(adaptPass2);

        MV_REGISTER_PASS(OptPass1)
        .setGenre(mv::PassGenre::Optimization)
        .setDescription("Test execution of a scheduled optimization pass")
        .setFunc(optPass1);

        MV_REGISTER_PASS(FinalPass1)
        .setGenre(mv::PassGenre::Finalization)
        .setDescription("Test execution of a scheduled finalization pass")
        .setFunc(finalPass1);

        MV_REGISTER_PASS(SerialPass1)
        .setGenre(mv::PassGenre::Serialization)
        .setDescription("Test execution of a scheduled serialization pass")
        .setFunc(serialPass1);

        MV_REGISTER_PASS(ValidPass1)
        .setGenre(mv::PassGenre::Validation)
        .setDescription("Test execution of a scheduled validation pass")
        .setFunc(validPass1);

    }
}


TEST(pass_manager, execution)
{

    mv::OpModel model;
    auto input = model.input(mv::Shape(1), mv::DType::Unknown, mv::Order::LastDimMajor);
    model.output(input);

    mv::PassManager pm;

    mv::TargetDescriptor desc;
    desc.setTarget(mv::Target::ma2480);
    desc.appendAdaptPass("AdaptPass1");
    desc.appendAdaptPass("AdaptPass2");
    desc.appendOptPass("OptPass1");
    desc.appendFinalPass("FinalPass1");
    desc.appendSerialPass("SerialPass1");
    desc.appendValidPass("ValidPass1");

    pm.initialize(model, desc);
    ASSERT_TRUE(pm.ready());
    
    while (!pm.completed())
    {
        pm.step();
    }

    ASSERT_TRUE(pm.completed());
    ASSERT_TRUE(model.getInput()->getAttr("adapt1").getContent<bool>());
    ASSERT_TRUE(model.getInput()->getAttr("adapt2").getContent<bool>());
    ASSERT_TRUE(model.getInput()->getAttr("opt1").getContent<bool>());
    ASSERT_TRUE(model.getInput()->getAttr("final1").getContent<bool>());
    ASSERT_TRUE(model.getInput()->getAttr("serial1").getContent<bool>());
    ASSERT_EQ(model.getInput()->getAttr("valid").getContent<unsigned>(), 3);

}

