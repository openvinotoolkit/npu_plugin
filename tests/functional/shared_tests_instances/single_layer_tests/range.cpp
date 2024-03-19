// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/range.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class RangeLayerTestCommon : public RangeLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    // Latest 'RangeLayerTest::SetUp' builds 'start,stop,step' as non-CONST inputs, thus unable to infer output shape.
    // So using older SetUp that builds CONST inputs.
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        float start, stop, step;
        tie(start, stop, step, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::Shape inShape{1};  // dummy ranked tensor
        auto params = std::make_shared<ov::op::v0::Parameter>(ngPrc, inShape);
        auto start_constant = std::make_shared<ov::op::v0::Constant>(ngPrc, ngraph::Shape{}, start);
        auto stop_constant = std::make_shared<ov::op::v0::Constant>(ngPrc, ngraph::Shape{}, stop);
        auto step_constant = std::make_shared<ov::op::v0::Constant>(ngPrc, ngraph::Shape{}, step);
        auto range = std::make_shared<ov::op::v4::Range>(start_constant, stop_constant, step_constant, ngPrc);
        const ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(range)};
        function = std::make_shared<ngraph::Function>(results, ov::ParameterVector{params}, "Range");
    }

    void Infer() override {
        // Cancel latest implementation, rely on default behavior
        LayerTestsCommon::Infer();
    }
};
class RangeLayerTest_NPU3720 : public RangeLayerTestCommon {};

TEST_P(RangeLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<InferenceEngine::Precision> InputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> OutputPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<float> start = {2.0f, 1.0f};
const std::vector<float> stop = {23.0f, 15.0f};
const std::vector<float> step = {3.0f, 4.5f};

const auto testRangePositiveStepParams = ::testing::Combine(
        testing::ValuesIn(start),  // start
        testing::ValuesIn(stop),   // stop
        testing::ValuesIn(step),   // positive step
        testing::ValuesIn(netPrecisions), testing::ValuesIn(InputPrecisions), testing::ValuesIn(OutputPrecisions),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto testRangeNegativeStepParams = ::testing::Combine(
        testing::Values(23.0f),  // start
        testing::Values(2.0f),   // stop
        testing::Values(-3.0f),  // negative step
        testing::ValuesIn(netPrecisions), testing::ValuesIn(InputPrecisions), testing::ValuesIn(OutputPrecisions),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// NPU3720
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Range, RangeLayerTest_NPU3720, testRangePositiveStepParams,
                         RangeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_negative_Range, RangeLayerTest_NPU3720, testRangeNegativeStepParams,
                         RangeLayerTest_NPU3720::getTestCaseName);

}  // namespace
