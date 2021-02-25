//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

// TODO: Temporarly skip this test for Ubuntu20 with gcc9 due to internal
// GCC compiler error (LTO) happening on CI infrastructure
#if __GNUC__ != 9

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/graphOptimizations/streaming_performace.hpp"
#include <file_utils.h>



// This fucnction builds a subgraph corresponding to chain 1 in Yolo v2
static void build_Function_0(mv::OpModel& model)
{
    const auto input = model.input("input", {13, 13, 1024, 1}, mv::DType("UInt8"), mv::Order("NHWC"), true);
    input->set<mv::QuantizationParams>("quantParams",  {{0},{1.0/255.0},{},{}});

    //Conv15 Yolo V2
    std::vector<int64_t> weights_conv15_data = mv::utils::generateSequence<int64_t> (1*1*512*1024,255, 0); 
    const auto conv15_weights = model.constantInt("conv15_weights", weights_conv15_data, {1, 1, 1024, 512}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv15_weights->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    const auto conv15 = model.conv("conv15", input, conv15_weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv15->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});

    //Conv16 Yolo V2
    std::vector<int64_t> weights_conv16_data = mv::utils::generateSequence<int64_t> (3*3*512*1024,255, 0); 
    const auto conv16_weights = model.constantInt("conv16_weights", weights_conv16_data, {3, 3, 512, 1024}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv16_weights->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    const auto conv16 = model.conv("conv16", conv15, conv16_weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv16->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});

    //Conv17 Yolo V2
    std::vector<int64_t> weights_conv17_data = mv::utils::generateSequence<int64_t> (1*1*512*1024,255, 0); 
    const auto conv17_weights = model.constantInt("conv17_weights", weights_conv17_data, {1, 1, 1024, 512}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv17_weights->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    const auto conv17 = model.conv("conv17", conv16, conv17_weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv17->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});

    //Conv18 Yolo V2
    std::vector<int64_t> weights_conv18_data = mv::utils::generateSequence<int64_t> (3*3*512*1024,255, 0); 
    const auto conv18_weights = model.constantInt("conv18_weights", weights_conv18_data, {3, 3, 512, 1024}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv18_weights->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    const auto conv18 = model.conv("conv18", conv17, conv18_weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv17->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});

    //higher_features/conv1 Yolo V2
    std::vector<int64_t> weights_conv1_data = mv::utils::generateSequence<int64_t> (3*3*1024*1024,255, 0); 
    const auto conv1_weights = model.constantInt("conv1_weights", weights_conv1_data, {3, 3, 1024, 1024}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv1_weights->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    const auto conv1 = model.conv("conv1", conv18, conv1_weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv1->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});

    //higher_features/conv2 Yolo V2
    std::vector<int64_t> weights_conv2_data = mv::utils::generateSequence<int64_t> (3*3*1024*1024,255, 0); 
    const auto conv2_weights = model.constantInt("conv2_weights", weights_conv2_data, {3, 3, 1024, 1024}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv2_weights->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    const auto conv2 = model.conv("conv2", conv1, conv2_weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv2->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});

    auto output = model.output("", conv2, mv::DType("UInt8"), true);
}


static void build_Function_1(mv::OpModel& model)
{
    const auto input = model.input("input", {13, 13, 1024, 1}, mv::DType("UInt8"), mv::Order("NHWC"), true);
    input->set<mv::QuantizationParams>("quantParams",  {{0},{1.0/255.0},{},{}});

    //higher_features/conv2 Yolo V2
    std::vector<int64_t> weights_conv2_data = mv::utils::generateSequence<int64_t> (3*3*1024*1024,255, 0); 
    const auto conv2_weights = model.constantInt("conv2_weights", weights_conv2_data, {3, 3, 1024, 1024}, mv::DType("UInt8"), mv::Order("NCHW"));
    conv2_weights->set<mv::QuantizationParams>("quantParams", {{0},{1.0/255.0},{},{}});
    const auto conv2 = model.conv("conv2", input, conv2_weights, {1, 1}, {0, 0, 0, 0}, 1, 1);
    conv2->set<mv::QuantizationParams>("quantParams", {{0},{144.0/255.0},{},{}});

    auto output = model.output("", conv2, mv::DType("UInt8"), true);
}

// This unit test is a system test. It tests that the StreamingPerformance pass increases the number of K streams
// for layer higher_features/conv2 to the maximum number of k streams that is possible for the layer.
// This is necessary to maximise performance for this network.
TEST(mcmCompiler, streaming_performance_system_test_yoloV2) {
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_Function_0(om);
    std::string compDescPath =
            InferenceEngine::getIELibraryPath() + "/mcm_config/compilation/release_kmb_with_CM_Conv.json";
    std::string targetDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/target/release_kmb.json";

    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "verbose", mv::Attribute(std::string("Info")));
    unit.loadTargetDescriptor(targetDescPath);
    unit.initialize();
    unit.run();

    std::shared_ptr<mv::Element> returnedParams = om.getGlobalConfigParams();
    auto streamingStrategyList = returnedParams->get<std::vector<mv::Element>>("streaming_strategy");

    for (auto layerNameStrategy : streamingStrategyList) {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");

        if (nodeName == "conv2") {
            auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
            auto kStream = streaming_strategy[3].get<int>("K");
            ASSERT_EQ(kStream, 16);
        }
    }
}

TEST(mcmCompiler, streaming_performance_per_cluster_weights_size) {

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_Function_1(om);
    std::string compDescPath =
            InferenceEngine::getIELibraryPath() + "/mcm_config/compilation/release_kmb_with_CM_Conv.json";
    std::string targetDescPath = InferenceEngine::getIELibraryPath() + "/mcm_config/target/release_kmb.json";

    unit.loadCompilationDescriptor(compDescPath);
    unit.compilationDescriptor().setPassArg("GlobalConfigParams", "verbose", mv::Attribute(std::string("Info")));
    unit.loadTargetDescriptor(targetDescPath);
    unit.initialize();
    unit.run();

    mv::StreamingPerformance streamingPerformance(om);
    std::shared_ptr<mv::Element> returnedParams = om.getGlobalConfigParams();
    auto streamingStrategyList = returnedParams->get<std::vector<mv::Element>>("streaming_strategy");
    auto multiClusterStrategyList = returnedParams->get<std::vector<mv::Element>>("split_strategy");
    mv::Attribute multiClusterStrategy;
    std::string mcStrategy;

    for (auto& opIt : om.getOps("DPUTask")) {
        mv::Data::OpListIterator oitr = om.getOp(opIt->getName());
        auto axisToSplit = opIt->get<unsigned>("axisToSplit");
        auto numberOfSplits = opIt->get<unsigned>("number_of_splits");
        auto multiClusterStrategy = opIt->get<std::string>("splitStrategy");

        ASSERT_EQ(axisToSplit, 3);
        size_t perClusterWeightsSize = streamingPerformance.calculateperClusterWeightsSize(
                *oitr, multiClusterStrategy, false, {1, 1, 1, numberOfSplits, 1});

        ASSERT_EQ(perClusterWeightsSize, 2359552);
    }
}

#endif // __GNUC__
