#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/base/jsonable.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

// Notes regarding the test: 2 model types are used below. A simple model without constants and another model which is more detailed to get control flow, groups etc.
// Memory allocators are not yet converted toJSON. 
// The tests run capture different attributes of computation model and checks if toJSON conversion is implemented correctly. 
// ground truths are hardcoded
// commented out test cases at the end may be obsolete (except memory allocators)

void setSimpleModel(mv::CompilationUnit& unit)
{
    
    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input({24, 24, 20, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto pool1It = cm.maxPool(input, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool2It = cm.maxPool(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    auto pool3It = cm.maxPool(pool1It, {1, 1}, {1, 1}, {0, 0, 0, 0});

    auto concat1It = cm.add({pool3It, pool2It});
    auto pool4It = cm.maxPool(concat1It, {1, 1}, {1, 1}, {0, 0, 0, 0});
    cm.output(pool4It);
}

void setModel(mv::CompilationUnit& unit)
{


    // Obtain compositional model from the compilation unit
    mv::OpModel& om = unit.model();
    // Initialize weights data
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> weights2Data = mv::utils::generateSequence<double>(5u * 5u * 8u * 16u);
    std::vector<double> weights3Data = mv::utils::generateSequence<double>(4u * 4u * 16u * 32u);
    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Info);

    // Compose model - use Composition API to create ops and obtain tensors
    auto input = om.input({128, 128, 3,1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto weights1 = om.constant(weights1Data, {3, 3, 3, 8}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv1 = om.conv(input, weights1, {2, 2}, {1, 1, 1, 1}, 1);
    auto pool1 = om.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = om.constant(weights2Data, {5, 5, 8, 16}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv2 = om.conv(pool1, weights2, {2, 2}, {2, 2, 2, 2}, 1);
    auto pool2 = om.maxPool(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = om.constant(weights3Data, {4, 4, 16, 32}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {0, 0, 0, 0}, 1);
    om.output(conv3);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2490))
    	exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from the compilation unit
    // Output DOT - file name (base)
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", false);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("GenerateBlob", "fileName", std::string("allocate_resources.blob"));
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // Initialize compilation
    unit.initialize();

    // Run all passes
    auto result = unit.run();

    // Obtain ops from tensors and add them to groups
    auto pool1Op = om.getSourceOp(pool1);
    auto pool2Op = om.getSourceOp(pool2);

    auto group1It = om.addGroup("pools");
    om.addGroupElement(pool1Op, group1It);
    om.addGroupElement(pool2Op, group1It);

    auto group2It = om.addGroup("convs");
    auto conv1Op = om.getSourceOp(conv1);
    auto conv2Op = om.getSourceOp(conv2);
    auto conv3Op = om.getSourceOp(conv3);
    om.addGroupElement(conv1Op, group2It);
    om.addGroupElement(conv2Op, group2It);
    om.addGroupElement(conv3Op, group2It);

    // Add groups to another group
    auto group3It = om.addGroup("ops");
    om.addGroupElement(group1It, group3It);
    om.addGroupElement(group2It, group3It);

    // Add ops that are already in some group to another group
    auto group4It = om.addGroup("first");
    om.addGroupElement(conv1Op, group4It);
    om.addGroupElement(pool1Op, group4It);

    mv::ControlModel cm(om);

    auto stage1It = cm.addStage();
    auto stage2It = cm.addStage();
    auto stage3It = cm.addStage();
    auto stage4It = cm.addStage();
    auto stage5It = cm.addStage();

    cm.addToStage(stage1It, conv1Op);
    cm.addToStage(stage2It, pool1Op);
    cm.addToStage(stage3It, conv2Op);
    cm.addToStage(stage4It, pool2Op);
    cm.addToStage(stage5It, conv3Op);

    cm.removeStage(stage5It);
}

TEST(jsonable, tensor)
{
    mv::Shape s({3, 3, 64,1});
    mv::Tensor t("test_tensor", s, mv::DType("Float16"), mv::Order("NCHW"));
    mv::json::Value v = t.toJSON();
    std::string result(v.stringify());
    std::string groundtruth = "{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[3,3,64,1]}},\"name\":\"test_tensor\"}";
    ASSERT_EQ(result, groundtruth);
}

TEST(jsonable, DISABLED_data_flow)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Array data_flow = unit.model().dataFlowToJSON();
//    std::cout << "This is data_flows: " << data_flow.stringify() << std::endl;
    std::string groundtruth = "[{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Input_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Input_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Input_00_Conv_00\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Constant_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":1,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Constant_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Constant_00_Conv_01\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Conv_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Conv_00_MaxPool_00\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_1\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_MaxPool_00_Conv_10\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Constant_1:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":1,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_1\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Constant_1\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Constant_10_Conv_11\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Conv_1:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_1\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Conv_10_MaxPool_10\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_2\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_MaxPool_10_Conv_20\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Constant_2:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":1,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_2\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Constant_2\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Constant_20_Conv_21\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Conv_2:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Output_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_2\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Conv_20_Output_00\"}]";
    ASSERT_EQ(groundtruth, data_flow.stringify());
}

TEST(jsonable, DISABLED_control_flow)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Array control_flow = unit.model().controlFlowToJSON();
//    std::cout << "This is control_flow: " << control_flow.stringify() << std::endl;
    std::string groundtruth = "[{\"attrs\":{\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Input_0\",\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"cf_Input_0_Conv_0\"},{\"attrs\":{\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_0\",\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"cf_Conv_0_MaxPool_0\"},{\"attrs\":{\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_1\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"cf_MaxPool_0_Conv_1\"},{\"attrs\":{\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_1\",\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"cf_Conv_1_MaxPool_1\"},{\"attrs\":{\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_2\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"cf_MaxPool_1_Conv_2\"},{\"attrs\":{\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Output_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_2\",\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"cf_Conv_2_Output_0\"}]";
    ASSERT_EQ(groundtruth, control_flow.stringify());
}


TEST(jsonable, DISABLED_stages)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Array stages = unit.model().stagesToJSON();
//    std::cout <<" This is the stages: "<< stages.stringify() << std::endl;
    std::string groundtruth = "[{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":0},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"Conv_0\"]}},\"model\":\"Model1\",\"name\":\"stage0\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":1},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"MaxPool_0\"]}},\"model\":\"Model1\",\"name\":\"stage1\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":2},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"Conv_1\"]}},\"model\":\"Model1\",\"name\":\"stage2\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":3},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"MaxPool_1\"]}},\"model\":\"Model1\",\"name\":\"stage3\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":4},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"Conv_2\"]}},\"model\":\"Model1\",\"name\":\"stage4\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":5},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"Conv_0\"]}},\"model\":\"Model1\",\"name\":\"stage5\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":6},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"MaxPool_0\"]}},\"model\":\"Model1\",\"name\":\"stage6\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":7},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"Conv_1\"]}},\"model\":\"Model1\",\"name\":\"stage7\"},{\"attrs\":{\"idx\":{\"attrType\":\"std::size_t\",\"content\":8},\"members\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"MaxPool_1\"]}},\"model\":\"Model1\",\"name\":\"stage8\"}]";
    ASSERT_EQ(groundtruth, stages.stringify());

}
TEST(jsonable, DISABLED_groups)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Array groups = unit.model().groupsToJSON();
//    std::cout <<" This is the groups: "<< groups.stringify() << std::endl;
    std::string groundtruth = "[{\"attrs\":{\"controlFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"dataFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"groups\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"ops\"]},\"ops\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"Conv_0\",\"Conv_1\",\"Conv_2\"]},\"stages\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"tensors\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]}},\"model\":\"Model1\",\"name\":\"convs\"},{\"attrs\":{\"controlFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"dataFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"groups\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"ops\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"Conv_0\",\"MaxPool_0\"]},\"stages\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"tensors\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]}},\"model\":\"Model1\",\"name\":\"first\"},{\"attrs\":{\"controlFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"dataFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"groups\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"pools\",\"convs\"]},\"ops\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"stages\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"tensors\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]}},\"model\":\"Model1\",\"name\":\"ops\"},{\"attrs\":{\"controlFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"dataFlows\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"groups\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"ops\"]},\"ops\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"MaxPool_0\",\"MaxPool_1\"]},\"stages\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]},\"tensors\":{\"attrType\":\"std::vector<std::string>\",\"content\":[]}},\"model\":\"Model1\",\"name\":\"pools\"}]";
    ASSERT_EQ(groundtruth, groups.stringify());

}
TEST(jsonable, DISABLED_opsIndexCounter)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Object opsIndexCounter = unit.model().opsIndexCounterToJSON();
//    std::cout <<" This is the opsIndexCounter: "<< opsIndexCounter.stringify() << std::endl;
    std::string groundtruth = "{\"Constant\":3.0,\"Conv\":3.0,\"Input\":1.0,\"MaxPool\":2.0,\"Output\":1.0}";
    ASSERT_EQ(groundtruth, opsIndexCounter.stringify());

}
TEST(jsonable, DISABLED_opsInstanceCounter)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Object opsInstanceCounter = unit.model().opsInstanceCounterToJSON();
//    std::cout <<" This is the opsInstanceCounter: "<< opsInstanceCounter.stringify() << std::endl;
    std::string groundtruth = "{\"Constant\":2.0,\"Conv\":2.0,\"Input\":0.0,\"MaxPool\":1.0,\"Output\":0.0}";
    ASSERT_EQ(groundtruth, opsInstanceCounter.stringify());

}

TEST(jsonable, DISABLED_opModelTensors)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Array tensors = unit.model().tensorsToJSON();
//    std::cout <<" This is the tensors: "<< tensors.stringify() << std::endl;
    std::string groundtruth = "[{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"ConstantMemory\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Constant_00_Conv_01\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"HWCN\"},\"populated\":{\"attrType\":\"bool\",\"content\":true},\"shape\":{\"attrType\":\"Shape\",\"content\":[3,3,3,8]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Constant_0\",\"traits\":[\"const\"]}},\"name\":\"Constant_0:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"ConstantMemory\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Constant_10_Conv_11\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"HWCN\"},\"populated\":{\"attrType\":\"bool\",\"content\":true},\"shape\":{\"attrType\":\"Shape\",\"content\":[5,5,8,16]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Constant_1\",\"traits\":[\"const\"]}},\"name\":\"Constant_1:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"ConstantMemory\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Constant_20_Conv_21\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"HWCN\"},\"populated\":{\"attrType\":\"bool\",\"content\":true},\"shape\":{\"attrType\":\"Shape\",\"content\":[4,4,16,32]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Constant_2\",\"traits\":[\"const\"]}},\"name\":\"Constant_2:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"IntermediateMemory\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Conv_00_MaxPool_00\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"HWC\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[64,64,8]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_0\",\"traits\":[\"const\"]}},\"name\":\"Conv_0:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"IntermediateMemory\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Conv_10_MaxPool_10\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"HWC\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[16,16,16]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_1\",\"traits\":[\"const\"]}},\"name\":\"Conv_1:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"ProgrammableOutput\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Conv_20_Output_00\"]},\"modelOutput\":{\"attrType\":\"bool\",\"content\":true},\"order\":{\"attrType\":\"Order\",\"content\":\"HWC\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[1,1,32]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Conv_2\",\"traits\":[\"const\"]}},\"name\":\"Conv_2:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"ProgrammableInput\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Input_00_Conv_00\"]},\"modelInput\":{\"attrType\":\"bool\",\"content\":true},\"order\":{\"attrType\":\"Order\",\"content\":\"HWC\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[128,128,3]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Input_0\",\"traits\":[\"const\"]}},\"name\":\"Input_0:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"IntermediateMemory\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_MaxPool_00_Conv_10\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"HWC\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[32,32,8]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]}},\"name\":\"MaxPool_0:0\"},{\"attrs\":{\"allocator\":{\"attrType\":\"std::string\",\"content\":\"IntermediateMemory\"},\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_MaxPool_10_Conv_20\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"HWC\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[4,4,16]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]}},\"name\":\"MaxPool_1:0\"}]";
    ASSERT_EQ(groundtruth, tensors.stringify());

}


TEST(jsonable, ops)
{
    mv::CompilationUnit unit("Model1");
    setSimpleModel(unit);
    mv::json::Array ops = unit.model().opsToJSON();
//    std::cout <<" This is the ops: "<< ops.stringify() << std::endl;
    std::string groundtruth = "[{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"opType\":{\"attrType\":\"std::string\",\"content\":\"Input\",\"traits\":[\"const\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"exposed\"]}},\"model\":\"Model1\",\"name\":\"Input_0\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_0\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_1\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_2\"},{\"attrs\":{\"opType\":{\"attrType\":\"std::string\",\"content\":\"Add\",\"traits\":[\"const\"]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"Add_0\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_3\"},{\"attrs\":{\"opType\":{\"attrType\":\"std::string\",\"content\":\"Output\",\"traits\":[\"const\"]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"exposed\"]}},\"model\":\"Model1\",\"name\":\"Output_0\"}]";
    ASSERT_EQ(groundtruth, ops.stringify());
    

}


TEST(jsonable, DISABLED_hasPopulatedTensors)
{
    mv::CompilationUnit unit("Model1");
    setModel(unit);
    mv::json::Bool hasPopulatedTensors = unit.model().hasPopulatedTensorsToJSON();
//    std::cout <<" This is the has populated tensors: "<< hasPopulatedTensors.stringify() << std::endl;
    std::string groundtruth = "true";
    ASSERT_EQ(groundtruth, hasPopulatedTensors.stringify());

}


TEST(jsonable, computationModel)
{
    mv::CompilationUnit unit("Model1");
    setSimpleModel(unit);
    mv::json::Value computationModel = unit.model().toJSON();
//  std::cout <<" This is the cm: "<< computationModel.stringify() << std::endl;
    std::string groundtruth = "{\"computation_groups\":[],\"graph\":{\"control_flows\":[],\"data_flows\":[{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Input_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Input_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Input_00_MaxPool_00\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_MaxPool_00_MaxPool_10\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_2\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_MaxPool_00_MaxPool_20\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_2:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Add_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_2\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_MaxPool_20_Add_00\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":1,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Add_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_MaxPool_10_Add_01\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"Add_0:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_3\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Add_0\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_Add_00_MaxPool_30\"},{\"attrs\":{\"data\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_3:0\",\"traits\":[\"const\"]},\"sinkInput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]},\"sinkOp\":{\"attrType\":\"std::string\",\"content\":\"Output_0\",\"traits\":[\"const\"]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_3\",\"traits\":[\"const\"]},\"sourceOutput\":{\"attrType\":\"std::size_t\",\"content\":0,\"traits\":[\"const\"]}},\"model\":\"Model1\",\"name\":\"df_MaxPool_30_Output_00\"}],\"nodes\":[{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"opType\":{\"attrType\":\"std::string\",\"content\":\"Input\",\"traits\":[\"const\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"exposed\"]}},\"model\":\"Model1\",\"name\":\"Input_0\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_0\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_1\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_2\"},{\"attrs\":{\"opType\":{\"attrType\":\"std::string\",\"content\":\"Add\",\"traits\":[\"const\"]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"Add_0\"},{\"attrs\":{,\"exclude_pad\":{\"attrType\":\"bool\",\"content\":true},\"kSize\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"opType\":{\"attrType\":\"std::string\",\"content\":\"MaxPool\",\"traits\":[\"const\"]},\"padding\":{\"attrType\":\"std::array<unsigned short, 4>\",\"content\":[0,0,0,0]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"stride\":{\"attrType\":\"std::array<unsigned short, 2>\",\"content\":[1,1]},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"executable\",\"exposed\"]}},\"model\":\"Model1\",\"name\":\"MaxPool_3\"},{\"attrs\":{\"opType\":{\"attrType\":\"std::string\",\"content\":\"Output\",\"traits\":[\"const\"]},\"quantParams\":{\"attrType\":\"mv::QuantizationParams\",\"content\":{\"attrs\":{\"max\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"min\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"scale\":{\"attrType\":\"std::vector<double>\",\"content\":[]},\"zeroPoint\":{\"attrType\":\"std::vector<int64_t>\",\"content\":[]}},\"name\":\"quantParams\"}},\"traits\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"exposed\"]}},\"model\":\"Model1\",\"name\":\"Output_0\"}]},\"has_populated_tensors\":false,\"operations_Index_counters\":{\"Add\":1.0,\"Input\":1.0,\"MaxPool\":4.0,\"Output\":1.0},\"operations_Instance_counters\":{\"Add\":1.0,\"Input\":1.0,\"MaxPool\":4.0,\"Output\":1.0},\"stages\":[],\"tensors\":[{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Add_00_MaxPool_30\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Add_0\",\"traits\":[\"const\"]}},\"name\":\"Add_0:0\"},{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_Input_00_MaxPool_00\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"Input_0\",\"traits\":[\"const\"]}},\"name\":\"Input_0:0\"},{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_MaxPool_00_MaxPool_10\",\"df_MaxPool_00_MaxPool_20\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_0\",\"traits\":[\"const\"]}},\"name\":\"MaxPool_0:0\"},{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_MaxPool_10_Add_01\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_1\",\"traits\":[\"const\"]}},\"name\":\"MaxPool_1:0\"},{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_MaxPool_20_Add_00\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_2\",\"traits\":[\"const\"]}},\"name\":\"MaxPool_2:0\"},{\"attrs\":{\"dType\":{\"attrType\":\"DType\",\"content\":\"Float16\"},\"flows\":{\"attrType\":\"std::set<std::string>\",\"content\":[\"df_MaxPool_30_Output_00\"]},\"order\":{\"attrType\":\"Order\",\"content\":\"NCHW\"},\"populated\":{\"attrType\":\"bool\",\"content\":false},\"shape\":{\"attrType\":\"Shape\",\"content\":[24,24,20,1]},\"sourceOp\":{\"attrType\":\"std::string\",\"content\":\"MaxPool_3\",\"traits\":[\"const\"]}},\"name\":\"MaxPool_3:0\"}]}";
    ASSERT_EQ(groundtruth, computationModel.stringify());
}

/*TEST(jsonable, int)
{
    int number = -1;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "-1");
}

TEST(jsonable, unsigned)
{
    unsigned number = 1;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "1");
}

TEST(jsonable, double)
{
    double number = 1.56;
    mv::json::Value v = mv::Jsonable::toJsonValue(number);
    std::string result(v.stringify());
    ASSERT_EQ(result, "1.56");
}

TEST(jsonable, vector4d)
{
    mv::Vector4D<double> vec;
    vec.e0 = 1.0;
    vec.e1 = 2.0;
    vec.e2 = 3.0;
    vec.e3 = 4.0;
    mv::json::Value v = mv::Jsonable::toJsonValue(vec);
    std::string result(v.stringify());
    ASSERT_EQ(result, "[1.0,2.0,3.0,4.0]");
}

TEST(jsonable, bool)
{
    bool true_value = true;
    mv::json::Value v = mv::Jsonable::toJsonValue(true_value);
    ASSERT_EQ(v.stringify(), "true");
    bool true_value_bis = mv::Jsonable::constructBoolTypeFromJson(v);
    ASSERT_EQ(true_value, true_value_bis);
    bool false_value = false;
    mv::json::Value v1 = mv::Jsonable::toJsonValue(false_value);
    ASSERT_EQ(v1.stringify(), "false");
    bool false_value_bis = mv::Jsonable::constructBoolTypeFromJson(v1);
    ASSERT_EQ(false_value, false_value_bis);
}

TEST(jsonable, attribute1)
{
    mv::Attribute att(mv::AttrType::DTypeType, mv::DTypeType::Float);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    ASSERT_EQ(result, "{\"attrType\":\"dtype\",\"content\":\"Float\"}");
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}

TEST(jsonable, attribute2)
{
    mv::Vector4D<double> vec;
    vec.e0 = 1.0;
    vec.e1 = 2.0;
    vec.e2 = 3.0;
    vec.e3 = 4.0;
    mv::Attribute att(mv::AttrType::FloatVec4DType, vec);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}

TEST(jsonable, attribute_bool)
{
    mv::Attribute att(mv::AttrType::BoolType, true);
    mv::json::Value v = mv::Jsonable::toJsonValue(att);
    std::string result(v.stringify());
    mv::Attribute att2 = mv::Attribute::JsonAttributeFactory(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(att2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}

TEST(jsonable, shape)
{
    mv::Shape s({3, 3, 64, 100});
    mv::json::Value v = mv::Jsonable::toJsonValue(s);
    std::string result(v.stringify());
    mv::Shape s1(v);
    mv::json::Value v1 = mv::Jsonable::toJsonValue(s1);
    std::string result1(v1.stringify());
    ASSERT_EQ(result1, result);
}
*/
/*
TEST(jsonable, operation)
{
    mv::op::Add op("add_test");
    mv::json::Value v = mv::Jsonable::toJsonValue(op);
    std::string result(v.stringify());
    mv::op::Add op2(v);
    mv::json::Value v2 = mv::Jsonable::toJsonValue(op2);
    std::string result2(v2.stringify());
    ASSERT_EQ(result, result2);
}
TEST(jsonable, memory_allocator)
{
    mv::MemoryAllocator m("test_allocator", 2048);
    mv::Shape s(3, 3, 64, 1);
    mv::Tensor t("test_tensor", s, mv::DType("Float16"), mv::Order("NCHW"));
    mv::Tensor t1("test_tensor1", s, mv::DType("Float16"), mv::Order("NCHW"));
    m.allocate(t, 0);
    m.allocate(t1, 0);
    mv::json::Value v = mv::Jsonable::toJsonValue(m);
    std::string result(v.stringify());
    ASSERT_EQ(result, "{\"max_size\":2048,\"name\":\"test_allocator\",\"states\":[{\"buffers\":[{\"layout\":\"plain\",\"lenght\":576,\"name\":\"test_tensor\",\"offset\":0},{\"layout\":\"plain\",\"lenght\":576,\"name\":\"test_tensor1\",\"offset\":576}],\"free_space\":896,\"stage\":0}]}");
}
*/
