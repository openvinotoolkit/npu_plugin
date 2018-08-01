#include "gtest/gtest.h"
#include "include/mcm/pass/pass_manager.hpp"
#include "include/mcm/computation/model/op_model.hpp"

static void setPassReg()
{

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> adaptPass1 = 
        [](mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "adapt1", mv::Attribute(mv::AttrType::BoolType, true));

    };

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> adaptPass2 = 
        [](mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "adapt2", mv::Attribute(mv::AttrType::BoolType, true));

    };

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> validPass1 =
        [](mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&) 
    {   
        mv::OpModel om(model);
        if (!om.getInput()->hasAttr("valid"))
            om.addAttr(om.getInput(), "valid", mv::Attribute(mv::AttrType::UnsignedType, 1U));
        else
        {
            auto attr = om.getInput()->getAttr("valid");
            attr.setContent<unsigned>(attr.getContent<unsigned>() + 1);
        }

    };

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> optPass1 = 
        [](mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "opt1", mv::Attribute(mv::AttrType::BoolType, true));

    };

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> finalPass1 =
        [](mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "final1", mv::Attribute(mv::AttrType::BoolType, true));

    };

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> serialPass1 = 
        [](mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "serial1", mv::Attribute(mv::AttrType::BoolType, true));

    };

    std::function<void(mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> passWithArg = 
        [](mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   

    };

    mv::pass::PassRegistry::instance().enter("__TEST_AdaptPass1")
    .setGenre(mv::PassGenre::Adaptation)
    .setDescription("Test execution of a scheduled adaptation pass")
    .setFunc(adaptPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_AdaptPass2")
    .setGenre(mv::PassGenre::Adaptation)
    .setDescription("Test execution of a scheduled adaptation pass")
    .setFunc(adaptPass2);

    mv::pass::PassRegistry::instance().enter("__TEST_OptPass1")
    .setGenre(mv::PassGenre::Optimization)
    .setDescription("Test execution of a scheduled optimization pass")
    .setFunc(optPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_FinalPass1")
    .setGenre(mv::PassGenre::Finalization)
    .setDescription("Test execution of a scheduled finalization pass")
    .setFunc(finalPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_SerialPass1")
    .setGenre(mv::PassGenre::Serialization)
    .setDescription("Test execution of a scheduled serialization pass")
    .setFunc(serialPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_ValidPass1")
    .setGenre(mv::PassGenre::Validation)
    .setDescription("Test execution of a scheduled validation pass")
    .setFunc(validPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_PassWithArg")
    .setGenre(mv::PassGenre::Adaptation)
    .setDescription("Test checking the required args in compilation decriptor")
    .setFunc(passWithArg)
    .defineArg(mv::json::JSONType::String, "arg1");

}

static void resetPassReg()
{
    mv::pass::PassRegistry::instance().remove("__TEST_AdaptPass1");
    mv::pass::PassRegistry::instance().remove("__TEST_AdaptPass2");
    mv::pass::PassRegistry::instance().remove("__TEST_OptPass1");
    mv::pass::PassRegistry::instance().remove("__TEST_FinalPass1");
    mv::pass::PassRegistry::instance().remove("__TEST_SerialPass1");
    mv::pass::PassRegistry::instance().remove("__TEST_ValidPass1");
    mv::pass::PassRegistry::instance().remove("__TEST_PassWithArg");
}


TEST(pass_manager, invalid_execution)
{

    setPassReg();
    mv::OpModel model;
    auto input = model.input(mv::Shape(1), mv::DType::Unknown, mv::Order::ColumnMajor);
    model.output(input);

    mv::PassManager pm;

    mv::TargetDescriptor targetDesc;
    mv::json::Object compDesc;

    ASSERT_FALSE(pm.ready());
    pm.initialize(model, targetDesc, compDesc);

    ASSERT_TRUE(pm.ready());
    ASSERT_FALSE(pm.validDescriptors());
    ASSERT_ANY_THROW(pm.step());

    targetDesc.setTarget(mv::Target::ma2480);
    targetDesc.setDType(mv::DType::Float);
    targetDesc.setOrder(mv::Order::ColumnMajor);
    pm.initialize(model, targetDesc, compDesc);
    pm.enablePass(mv::PassGenre::Adaptation, "__TEST_PassWithArg");
    ASSERT_TRUE(pm.ready());
    ASSERT_FALSE(pm.validDescriptors());
    ASSERT_ANY_THROW(pm.step());

    compDesc["arg1"] = std::string("value");
    pm.initialize(model, targetDesc, compDesc);
    ASSERT_TRUE(pm.ready());
    ASSERT_TRUE(pm.validDescriptors());
    resetPassReg();

}


TEST(pass_manager, execution)
{

    setPassReg();
    mv::OpModel model;
    auto input = model.input(mv::Shape(1), mv::DType::Unknown, mv::Order::ColumnMajor);
    model.output(input);

    mv::PassManager pm;

    mv::TargetDescriptor targetDesc;
    mv::json::Object compDesc;
    targetDesc.setTarget(mv::Target::ma2480);
    targetDesc.setDType(mv::DType::Float);
    targetDesc.setOrder(mv::Order::ColumnMajor);
    targetDesc.appendAdaptPass("__TEST_AdaptPass1");
    targetDesc.appendAdaptPass("__TEST_AdaptPass2");
    targetDesc.appendOptPass("__TEST_OptPass1");
    targetDesc.appendFinalPass("__TEST_FinalPass1");
    targetDesc.appendSerialPass("__TEST_SerialPass1");
    targetDesc.appendValidPass("__TEST_ValidPass1");

    pm.initialize(model, targetDesc, compDesc);
    ASSERT_TRUE(pm.ready());
    ASSERT_TRUE(pm.validDescriptors());
    
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
    ASSERT_EQ(model.getInput()->getAttr("valid").getContent<unsigned>(), 4);
    resetPassReg();

}

