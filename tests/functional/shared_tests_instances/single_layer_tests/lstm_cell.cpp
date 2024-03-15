// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/lstm_cell.hpp"
#include "vpu_ov1_layer_test.hpp"

using ngraph::helpers::InputLayerType;

namespace LayerTestsDefinitions {

class LSTMCellLayerTest_NPU3700 : public LSTMCellTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

TEST_P(LSTMCellLayerTest_NPU3700, HW) {
    threshold = 0.06;
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

class LSTMCellLayerTestCommon : public LSTMCellTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
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
        InputLayerType WType;
        InputLayerType RType;
        InputLayerType BType;
        InferenceEngine::Precision netPrecision;
        std::tie(should_decompose, batch, hidden_size, input_size, activations, clip, WType, RType, BType, netPrecision,
                 targetDevice) = this->GetParam();
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, input_size},
                 {batch, hidden_size},
                 {batch, hidden_size},
                 {4 * hidden_size, input_size},
                 {4 * hidden_size, hidden_size},
                 {4 * hidden_size}},
        };
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<ov::Shape> shapes{inputShapes[0], inputShapes[1], inputShapes[2],
                                      inputShapes[3], inputShapes[4], inputShapes[5]};
        ov::ParameterVector params;
        for (auto&& shape : shapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, shape));
        }
        ASSERT_EQ(InputLayerType::CONSTANT, WType);
        ASSERT_EQ(InputLayerType::CONSTANT, RType);
        ASSERT_EQ(InputLayerType::CONSTANT, BType);
        std::vector<ngraph::Shape> WRB = {inputShapes[3], inputShapes[4], inputShapes[5]};
        auto lstm_cell =
                ngraph::builder::makeLSTM(ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params)),
                                          WRB, hidden_size, activations, {}, {}, clip, false);
        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(lstm_cell->output(0)),
                                     std::make_shared<ov::op::v0::Result>(lstm_cell->output(1))};
        function = std::make_shared<ngraph::Function>(results, params, "lstm_cell");
    }
};

class LSTMCellLayerTest_NPU3720 : public LSTMCellLayerTestCommon {};

TEST_P(LSTMCellLayerTest_NPU3720, HW) {
    threshold = 0.06;
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
std::vector<bool> should_decompose{false};
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

// NPU3700 tests
std::vector<size_t> batch3700{1};
std::vector<size_t> hidden_size3700{1};
std::vector<size_t> input_size3700{5};

// Test scope was reduced to one test due to accuracy drop in many cases (accuracy deviation ~0.2(5%) from reference)
// that can't be moved to separate scope due to dissimilar parameters
// Also some simple test cases with small dimensions takes more than 5 sec time (don't fit to timeout)

INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon, LSTMCellLayerTest_NPU3700,
                        ::testing::Combine(::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch3700),
                                           ::testing::ValuesIn(hidden_size3700), ::testing::ValuesIn(input_size3700),
                                           ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
                                           ::testing::Values(InputLayerType::CONSTANT),
                                           ::testing::Values(InputLayerType::CONSTANT),
                                           ::testing::Values(InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        LSTMCellLayerTest_NPU3700::getTestCaseName);

// NPU3720
std::vector<size_t> batch{1};
std::vector<size_t> hidden_size{4, 64};
std::vector<size_t> input_size{6, 24};
std::vector<size_t> hidden_size_precommit{2, 16};
std::vector<size_t> input_size_precommit{3, 12};

const auto lstmCellConfig = ::testing::Combine(
        ::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hidden_size),
        ::testing::ValuesIn(input_size), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto lstmCellPrecommitConfig = ::testing::Combine(
        ::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hidden_size_precommit),
        ::testing::ValuesIn(input_size_precommit), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_LSTMCellCommon, LSTMCellLayerTest_NPU3720, lstmCellPrecommitConfig,
                        LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon, LSTMCellLayerTest_NPU3720, lstmCellConfig, LSTMCellTest::getTestCaseName);

}  // namespace
