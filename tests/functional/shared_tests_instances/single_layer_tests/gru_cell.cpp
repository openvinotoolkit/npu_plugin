//
// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/gru_cell.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class GRUCellLayerTestCommon : public GRUCellTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class GRUCellLayerTest_NPU3720 : public GRUCellLayerTestCommon {
    void SetUp() override {
        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
        GRUCellTest::SetUp();
    }
};

TEST_P(GRUCellLayerTest_NPU3720, HW) {
    threshold = 0.06;
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<bool> shouldDecompose{false};
const std::vector<size_t> batch{2};
const std::vector<size_t> hiddenSize{4};
const std::vector<size_t> inputSize{3};
const std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}};
const std::vector<float> clip{0.f};
const std::vector<bool> shouldLinearBeforeReset{true, false};
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};
const std::vector<ngraph::helpers::InputLayerType> WRBLayerTypes = {ngraph::helpers::InputLayerType::CONSTANT};

const auto gruCellParams = testing::Combine(
        ::testing::ValuesIn(shouldDecompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hiddenSize),
        ::testing::ValuesIn(inputSize), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::ValuesIn(shouldLinearBeforeReset), ::testing::ValuesIn(WRBLayerTypes),
        ::testing::ValuesIn(WRBLayerTypes), ::testing::ValuesIn(WRBLayerTypes), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_GRUCell, GRUCellLayerTest_NPU3720, gruCellParams, GRUCellTest::getTestCaseName);

}  // namespace
