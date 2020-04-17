#include "gtest/gtest.h"
#include "include/mcm/pass/pass_manager.hpp"
#include "include/mcm/op_model.hpp"

static void setPassReg()
{

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)> adaptPass1 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "adapt1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)> adaptPass2 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "adapt2", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)> validPass1 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {   
        mv::OpModel om(model);
        if (!om.getInput()->hasAttr("valid"))
            om.addAttr(om.getInput(), "valid", (std::size_t)1);
        else
            om.getInput()->set<std::size_t>("valid", om.getInput()->get<std::size_t>("valid") + 1);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)> optPass1 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "opt1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)> finalPass1 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "final1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)> serialPass1 =
        [](const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {   
        mv::OpModel om(model);
        om.addAttr(om.getInput(), "serial1", (bool)true);

    };

    std::function<void(const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)> passWithArg =
        [](const mv::pass::PassEntry&, mv::ComputationModel&, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {   

    };

    mv::pass::PassRegistry::instance().enter("__TEST_AdaptPass1")
    .setDescription("Test execution of a scheduled adaptation pass")
    .setFunc(adaptPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_AdaptPass2")
    .setDescription("Test execution of a scheduled adaptation pass")
    .setFunc(adaptPass2);

    mv::pass::PassRegistry::instance().enter("__TEST_OptPass1")
    .setDescription("Test execution of a scheduled optimization pass")
    .setFunc(optPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_FinalPass1")
    .setDescription("Test execution of a scheduled finalization pass")
    .setFunc(finalPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_SerialPass1")
    .setDescription("Test execution of a scheduled serialization pass")
    .setFunc(serialPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_ValidPass1")
    .setDescription("Test execution of a scheduled validation pass")
    .setFunc(validPass1);

    mv::pass::PassRegistry::instance().enter("__TEST_PassWithArg")
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
    auto input = model.input({1}, mv::DType("Float16"), mv::Order("W"));
    model.output(input);

    mv::PassManager pm;

    mv::TargetDescriptor targetDesc;
    mv::CompilationDescriptor compDesc;

    ASSERT_FALSE(pm.initialized());
    pm.initialize(model, targetDesc, compDesc);

    ASSERT_TRUE(pm.initialized());
    ASSERT_FALSE(pm.validDescriptors());
    ASSERT_ANY_THROW(pm.step());

    targetDesc.setTarget(mv::Target::ma2490);
    targetDesc.setDType(mv::DType("Float16"));

    // Initialize the compilation descriptor, but leave out pass argument.
    compDesc.addGroup("Adaptation");
    compDesc.addToGroup("Adaptation", "__TEST_PassWithArg", "Singular", false);
    compDesc.addGroup("root");
    compDesc.addToGroup("root", "Adaptation", "Singular", true);
    std::vector<mv::Element> passList = compDesc.serializePassList();
    pm.loadPassList(passList);

    ASSERT_TRUE(pm.initialized());
    // Missing pass argument should be caught here.
    ASSERT_FALSE(pm.validDescriptors());
    ASSERT_ANY_THROW(pm.step());

    mv::Attribute v1 = std::string("value");
    compDesc.setPassArg("__TEST_PassWithArg", "arg1", v1);

    pm.initialize(model, targetDesc, compDesc);
    passList = compDesc.serializePassList();
    pm.loadPassList(passList);

    ASSERT_TRUE(pm.initialized());
    ASSERT_TRUE(pm.validDescriptors());
    ASSERT_NO_THROW(pm.step());

    resetPassReg();

}


TEST(pass_manager, DISABLED_execution)
{

    setPassReg();
    mv::OpModel model("testModel");
    auto input = model.input({1}, mv::DType("Float16"), mv::Order("W"));
    model.output(input);

    mv::PassManager pm;

    mv::TargetDescriptor targetDesc;
    mv::CompilationDescriptor compDesc;
    targetDesc.setTarget(mv::Target::ma2490);
    targetDesc.setDType(mv::DType("Float16"));

    compDesc.addGroup("Adaptation");
    compDesc.addGroup("Optimization");
    compDesc.addGroup("Finalization");
    compDesc.addGroup("Serialization");
    compDesc.addGroup("Validation");
    compDesc.addToGroup("Adaptation", "__TEST_AdaptPass1", "Singular", false);
    compDesc.addToGroup("Adaptation", "__TEST_AdaptPass2", "Singular", false);
    compDesc.addToGroup("Optimization", "__TEST_OptPass1", "Singular", false);
    compDesc.addToGroup("Finalization", "__TEST_FinalPass1", "Singular", false);
    compDesc.addToGroup("Serialization", "__TEST_SerialPass1", "Singular", false);
    compDesc.addToGroup("Validation", "__TEST_ValidPass1", "Singular", false);

    compDesc.addGroup("root");
    compDesc.addToGroup("root", "Adaptation", "Singular", true);
    compDesc.addToGroup("root", "Optimization", "Singular", true);
    compDesc.addToGroup("root", "Finalization", "Singular", true);
    compDesc.addToGroup("root", "Serialization", "Singular", true);
    compDesc.addToGroup("root", "Validation", "Recurrent", true);

    pm.initialize(model, targetDesc, compDesc);

    std::vector<mv::Element> passList = compDesc.serializePassList();
    pm.loadPassList(passList);

    ASSERT_TRUE(pm.initialized());
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

