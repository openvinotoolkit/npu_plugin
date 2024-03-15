//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include <gtest/gtest.h>

#include "zero_compiler_in_driver.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>

namespace {

struct NodeDescriptor {
    std::string name;
    ov::element::Type_t precision;
    ov::Shape shape;
};

}  // namespace

namespace vpux {
namespace driverCompilerAdapter {

class ZeroCompilerAdapterTests : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> createModel(std::vector<NodeDescriptor> inputNodeDescriptors,
                                           std::vector<NodeDescriptor> outputNodeDescriptors) {
        ov::ParameterVector parameterVector;
        ov::NodeVector constantVector;

        for (const NodeDescriptor& nodeDescriptor : inputNodeDescriptors) {
            std::shared_ptr<ov::op::v0::Parameter> parameter =
                    std::make_shared<ov::op::v0::Parameter>(nodeDescriptor.precision, nodeDescriptor.shape);
            parameter->set_friendly_name(nodeDescriptor.name);
            parameterVector.push_back(std::move(parameter));
        }
        for (const NodeDescriptor& nodeDescriptor : outputNodeDescriptors) {
            std::shared_ptr<ov::Node> constant =
                    std::make_shared<ov::op::v0::Constant>(nodeDescriptor.precision, nodeDescriptor.shape);
            constant->set_friendly_name(nodeDescriptor.name);
            constantVector.push_back(std::move(constant));
        }

        return std::make_shared<ov::Model>(constantVector, parameterVector);
    }
};

TEST_F(ZeroCompilerAdapterTests, SingleIONetwork_ipU8opFP32) {
    const std::shared_ptr<ov::Model> model = createModel({{"inputName1", ov::element::Type_t::u8, {1, 1, 10, 100}}},
                                                         {{"outputName1", ov::element::Type_t::f32, {1, 100}}});
    const std::string ioInfo = LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::serializeIOInfo(model);

    const std::string expectedStr = "--inputs_precisions=\"inputName1:U8\" --inputs_layouts=\"inputName1:NCHW\""
                                    " --outputs_precisions=\"outputName1:FP32\" --outputs_layouts=\"outputName1:NC\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

TEST_F(ZeroCompilerAdapterTests, TwoIONetwork_ipU8U8opFP32FP32) {
    const std::shared_ptr<ov::Model> model = createModel(
            {{"inputName1", ov::element::Type_t::u8, {1, 1, 10, 100}},
             {"inputName2", ov::element::Type_t::u8, {1, 1, 10, 100}}},
            {{"outputName1", ov::element::Type_t::f32, {1, 100}}, {"outputName2", ov::element::Type_t::f32, {1, 100}}});

    const std::string ioInfo = LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::serializeIOInfo(model);

    const std::string expectedStr =
            "--inputs_precisions=\"inputName1:U8 inputName2:U8\" --inputs_layouts=\"inputName1:NCHW inputName2:NCHW\""
            " --outputs_precisions=\"outputName1:FP32 outputName2:FP32\" --outputs_layouts=\"outputName1:NC"
            " outputName2:NC\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

TEST_F(ZeroCompilerAdapterTests, OneInputTwoOuputsNetwork_ipU8opFP16FP32) {
    const std::shared_ptr<ov::Model> model = createModel({{"inputName1", ov::element::Type_t::u8, {1, 1, 10, 100}}},
                                                         {{"outputName1", ov::element::Type_t::f32, {1, 100}},
                                                          {"outputName2", ov::element::Type_t::f16, {1, 1, 10, 100}}});

    const std::string ioInfo = LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::serializeIOInfo(model);

    const std::string expectedStr = "--inputs_precisions=\"inputName1:U8\" --inputs_layouts=\"inputName1:NCHW\""
                                    " --outputs_precisions=\"outputName1:FP32 outputName2:FP16\""
                                    " --outputs_layouts=\"outputName1:NC outputName2:NCHW\"";
    EXPECT_EQ(ioInfo, expectedStr);
}

}  // namespace driverCompilerAdapter
}  // namespace vpux
