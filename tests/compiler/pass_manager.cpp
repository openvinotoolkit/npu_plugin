#include "gtest/gtest.h"
#include "include/mcm/pass/pass_manager.hpp"
#include "meta/include/mcm/op_model.hpp"

static void setPassReg()
{

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> adaptPass1 = 
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "adapt1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> adaptPass2 = 
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "adapt2", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> validPass1 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&) 
    {   
        mv::OpModel om(model);
        if (!om.getInput()->hasAttr("valid"))
            om.addAttr(om.getInput(), "valid", (std::size_t)1);
        else
            om.getInput()->set<std::size_t>("valid", om.getInput()->get<std::size_t>("valid") + 1);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> optPass1 = 
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "opt1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> finalPass1 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "final1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> serialPass1 = 
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "serial1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)> passWithArg = 
        [](const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
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
    mv::OpModel model("testModel");
    auto input = model.input({1}, mv::DTypeType::Float16, mv::Order("W"));
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
    targetDesc.setDType(mv::DTypeType::Float16);
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
    mv::OpModel model("testModel");
    auto input = model.input({1}, mv::DTypeType::Float16, mv::Order("W"));
    model.output(input);

    mv::PassManager pm;

    mv::TargetDescriptor targetDesc;
    mv::json::Object compDesc;
    targetDesc.setTarget(mv::Target::ma2480);
    targetDesc.setDType(mv::DTypeType::Float16);
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
    ASSERT_TRUE(model.getInput()->get<bool>("adapt1"));
    ASSERT_TRUE(model.getInput()->get<bool>("adapt2"));
    ASSERT_TRUE(model.getInput()->get<bool>("opt1"));
    ASSERT_TRUE(model.getInput()->get<bool>("final1"));
    ASSERT_TRUE(model.getInput()->get<bool>("serial1"));
    ASSERT_EQ(model.getInput()->get<std::size_t>("valid"), 4);
    resetPassReg();

}

