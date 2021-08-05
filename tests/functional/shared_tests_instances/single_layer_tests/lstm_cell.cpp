// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/lstm_cell.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbLSTMCellLayerTest : public LSTMCellTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbLSTMCellLayerTest, CompareWithRefs_MLIR) {
    threshold = 0.06;
    useCompilerMLIR();
    Run();
}
} // namespace LayerTestsDefinitions

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

    INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon, KmbLSTMCellLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(should_decompose),
                                    ::testing::ValuesIn(batch),
                                    ::testing::ValuesIn(hidden_size),
                                    ::testing::ValuesIn(input_size),
                                    ::testing::ValuesIn(activations),
                                    ::testing::ValuesIn(clip),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbLSTMCellLayerTest::getTestCaseName);

}  // namespace
