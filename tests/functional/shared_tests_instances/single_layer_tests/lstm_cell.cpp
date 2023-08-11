// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/lstm_cell.hpp"

namespace LayerTestsDefinitions {

class VPUXLSTMCellLayerTest : public LSTMCellTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXLSTMCellLayerTest_VPU3700 : public VPUXLSTMCellLayerTest {};

TEST_P(VPUXLSTMCellLayerTest_VPU3700, HW) {
    threshold = 0.06;
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

class VPUXLSTMCellLayerTest_VPU3720 : public VPUXLSTMCellLayerTest {
    void SetUp() override {
        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
        bool should_decompose;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        InferenceEngine::Precision netPrecision;
        std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, netPrecision, targetDevice) =
                this->GetParam();
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, input_size},
                 {batch, hidden_size},
                 {batch, hidden_size},
                 {4 * hidden_size, input_size},
                 {4 * hidden_size, hidden_size},
                 {4 * hidden_size}},
        };
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1], inputShapes[2],
                                                          inputShapes[3], inputShapes[4], inputShapes[5]});
        std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5]};
        auto lstm_cell =
                ngraph::builder::makeLSTM(ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)),
                                          WRB, hidden_size, activations, {}, {}, clip, false);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(lstm_cell->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(lstm_cell->output(1))};
        function = std::make_shared<ngraph::Function>(results, params, "lstm_cell");
    }
};

TEST_P(VPUXLSTMCellLayerTest_VPU3720, HW) {
    threshold = 0.06;
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose{false};
std::vector<size_t> batch{1};
std::vector<size_t> hidden_size{1};
std::vector<size_t> input_size{5};
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

// Test scope was reduced to one test due to accuracy drop in many cases (accuracy deviation ~0.2(5%) from reference)
// that can't be moved to separate scope due to dissimilar parameters
// Also some simple test cases with small dimensions takes more than 5 sec time (don't fit to timeout)

INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon, VPUXLSTMCellLayerTest_VPU3700,
                        ::testing::Combine(::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch),
                                           ::testing::ValuesIn(hidden_size), ::testing::ValuesIn(input_size),
                                           ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXLSTMCellLayerTest_VPU3700::getTestCaseName);

// VPU3720 test
std::vector<size_t> batch_VPU3720{1};
std::vector<size_t> hidden_size_VPU3720{4, 64};
std::vector<size_t> input_size_VPU3720{6, 24};
INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon_VPU3720, VPUXLSTMCellLayerTest_VPU3720,
                        ::testing::Combine(::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch_VPU3720),
                                           ::testing::ValuesIn(hidden_size_VPU3720),
                                           ::testing::ValuesIn(input_size_VPU3720), ::testing::ValuesIn(activations),
                                           ::testing::ValuesIn(clip), ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        LSTMCellTest::getTestCaseName);

std::vector<size_t> hidden_size_VPU3720_precomit{2, 16};
std::vector<size_t> input_size_VPU3720_precomit{3, 12};
INSTANTIATE_TEST_CASE_P(smoke_precommit_LSTMCellCommon_VPU3720, VPUXLSTMCellLayerTest_VPU3720,
                        ::testing::Combine(::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch_VPU3720),
                                           ::testing::ValuesIn(hidden_size_VPU3720_precomit),
                                           ::testing::ValuesIn(input_size_VPU3720_precomit),
                                           ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        LSTMCellTest::getTestCaseName);

}  // namespace
