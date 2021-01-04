// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reverse_sequence.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbReverseSequenceLayerTest: public ReverseSequenceLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    };

    TEST_P(KmbReverseSequenceLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::U8,
            // InferenceEngine::Precision::I8,  - This precision is not supported by KMB
            // InferenceEngine::Precision::U16, - This precision is not supported by KMB
            // InferenceEngine::Precision::I32  - This precision is not supported by KMB
    };

    const std::vector<int64_t> batchAxisIndices = { 0L };

    const std::vector<int64_t> seqAxisIndices = { 1L };

    const std::vector<std::vector<size_t>> inputShapes = { {3, 10} }; //, 10, 20

    const std::vector<std::vector<size_t>> reversSeqLengthsVecShapes = { {3} };

    const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
            ngraph::helpers::InputLayerType::CONSTANT,
            ngraph::helpers::InputLayerType::PARAMETER
    };

    // Test fails with error:
    // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration) doesn't throw
    // an exception.
    // Actual: it throws:Input data type is not supported: I32
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:212
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64
    // [Track number: S#45145]
    INSTANTIATE_TEST_CASE_P(DISABLED_Basic_smoke, KmbReverseSequenceLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(batchAxisIndices),
                                    ::testing::ValuesIn(seqAxisIndices),
                                    ::testing::ValuesIn(inputShapes),
                                    ::testing::ValuesIn(reversSeqLengthsVecShapes),
                                    ::testing::ValuesIn(secondaryInputTypes),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            KmbReverseSequenceLayerTest::getTestCaseName);

}  // namespace
