#include "mcm/tensor/shape.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/tensor/dtype/dtype.hpp"
#include "mcm/target/target_descriptor.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <string>
#include <unordered_set>


// TEST(workloads_metis, 45x45_mode_4_4_workloads_5)
// {
//     mv::CompilationUnit unit("45x45_mode_4_4_workloads_5");
//     mv::OpModel& om = unit.model();

//      auto input = om.input({45, 45, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}});
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*64*64);
//     auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}});
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}});
//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//      std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));

// }


// TEST(workloads_metis, 70x70_mode_4_4_workloads_5)
// {
//     mv::CompilationUnit unit("70x70_mode_4_4_workloads_");
//     mv::OpModel& om = unit.model();

//      auto input = om.input({70, 70, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}});
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*64*64);
//     auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}});
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}});
//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//      std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));

// }

// TEST(workloads_metis, 66x66_mode_4_4_workloads_5)
// {
//     mv::CompilationUnit unit("66x66_mode_4_4_workloads_5");
//     mv::OpModel& om = unit.model();

//     auto input = om.input({66, 66, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}});
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*64*64);
//     auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}});
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}});
//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//      std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));

// }


// TEST(workloads_metis, 62x62_mode_4_4_workloads_5)
// {
//     mv::CompilationUnit unit("62x62_mode_4_4_workloads_5");
//     mv::OpModel& om = unit.model();

//      auto input = om.input({62, 62, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}});
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*64*64);
//     auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}});
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}});
//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//      std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));

// }



// TEST(workloads_metis, 58x58_mode_4_4_workloads_5)
// {
//     mv::CompilationUnit unit("58x58_mode_4_4_workloads_5");
//     mv::OpModel& om = unit.model();

//      auto input = om.input({58, 58, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}});
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*64*64);
//     auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}});
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}});
//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//      std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));
// }



// TEST(workloads_metis, res3a_branch2c)
// {
//     mv::CompilationUnit unit("res3a_branch2c");
//     mv::OpModel& om = unit.model();

//      auto input = om.input({28, 28, 128, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*128*512);
//     auto weights = om.constantInt(weightsData, {1, 1, 128, 512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "res3a_branch2c_weights");
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}}, "res3a_branch2c#4");
//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//     std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));
// }

// TEST(workloads_metis, res2a_branch2a)
// {
//     mv::CompilationUnit unit("res2a_branch2a");
//     mv::OpModel& om = unit.model();

//     auto input = om.input({56, 56, 64, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*64*64);
//     auto weights = om.constantInt(weightsData, {1, 1, 64, 64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "res2a_branch2a_weights");
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}}, "res2a_branch2a#4");
//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//     std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));
// }

// TEST(workloads_metis, res4a_branch2b)
// {
//     mv::CompilationUnit unit("res4a_branch2b");
//     mv::OpModel& om = unit.model();

//     auto input = om.input({14, 14, 1024, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*1024*256);
//     auto weights = om.constantInt(weightsData, {1, 1, 1024, 256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "res4a_branch2b_weights");
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}}, "res4a_branch2b#4");

//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//     std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));
// }

// TEST(workloads_metis, res5a_branch2a)
// {
//     mv::CompilationUnit unit("res5a_branch2a");
//     mv::OpModel& om = unit.model();

//     auto input = om.input({7, 7, 100, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");
//     std::vector<int64_t> weightsData = mv::utils::generateSequence<int64_t>(1*1*100*512);
//     auto weights = om.constantInt(weightsData, {1, 1, 100, 512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{}, {}, {}, {}}, "res5a_branch2a_weights");
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0}, 1, 1, {{},{},{},{}}, "res5a_branch2a#4");

//     om.output(conv);

//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

//      std::string optString = "Metis";
//     mv::Attribute option = optString;
//     compDesc.setPassArg("GenerateWorkloads", "Metis", option);

//     unit.compilationDescriptor().remove("finalize","MaxTopologicalCutAndPartialSerialisation");
//     unit.compilationDescriptor().remove("serialize");

//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     unit.initialize();
//     unit.run();

//     auto convOp = om.getOps("DPUTask");
//     auto op = convOp[0];

//     EXPECT_TRUE(op->get<bool>("Valid_workload"));
// }




