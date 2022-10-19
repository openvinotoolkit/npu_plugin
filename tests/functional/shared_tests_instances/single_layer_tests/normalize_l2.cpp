//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/normalize_l2.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbNormalizeL2LayerTest: public NormalizeL2LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

    TEST_P(KmbNormalizeL2LayerTest, CompareWithRefs) {
        threshold = 0.04;
        Run();
    }
    TEST_P(KmbNormalizeL2LayerTest, CompareWithRefs_MLIR) {
        threshold = 0.04;
        useCompilerMLIR();
        Run();
    }
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

//
// If shape N-dimensional and axes contains N-1 dims - tests failed. [Track number: E#21695]
//
    const std::vector<std::vector<int64_t>> axes = {
            // {}, 16900: NormalizeL2 output mismatch for empty axes case
            {1},
            //{1,2}
            // Only for shapes = { ., ., .}
            //{0, 1, 2},
            //{0, 1, 2, 3}
    };

    // Values from real neural networks
    const std::vector<float> eps = {1.000000013351432e-10, 9.999999960041972e-13};

    //
    // Contains eps > threshold. Incorrect kernel work, because the "max" mode isn't supported. [Track number: E#21695]
    //
    //const std::vector<float> eps = {0.0001, 0.001, 0.5};


    const std::vector<ngraph::op::EpsMode> epsMode = {
            ngraph::op::EpsMode::ADD,
            ngraph::op::EpsMode::MAX,
    };

  // There is a error at validation step for shapes which size are not equal to 4.
    // Possibly it is error in run-time due to only 4D shapes are allowed.
    // Example of output on KMB-board:
    // ...
    // TestReportProgress: KmbNormalizeL2LayerTest inferred
    // KmbLayerTestsCommon::Validate()
    // LayerTestsCommon::Validate()
    // openvino/inference-engine/tests/functional/shared_test_classes/include/shared_test_classes/
    // base/layer_test_utils.hpp:173: Failure
    // Value of: max != 0 && (diff <= static_cast<float>(threshold))
    // Actual: false
    // Expected: true
    // Relative comparison of values expected: 0 and actual: nan at index 0 with threshold 0.0099999997764825821 failed
    // [Track number: S#52943]

//
//[Track number: E#21695]
//
    std::vector<std::vector<size_t>> shapes = {
            {1, 128},
            {1, 512},
            {1, 8, 24, 64},
            //{1, 3, 10, 5},

            // Turn off the axes = {0, 1, 2, 3}
            // Incorrect kernel work in case axes = {1}
            //
            //{1, 5, 3}

            // Values from real neural networks
            //{1, 512, 40, 40},
            //{1, 512, 20, 20},
            {1, 512, 64, 64}
            //{1, 512, 38, 38},
            //{1, 128, 25, 43},
            //{1, 128, 50, 85}

            //Incorrect kernel work
            //{1, 1, 1, 10}
    };

    const auto normL2params = testing::Combine(
            testing::ValuesIn(axes),
            testing::ValuesIn(eps),
            testing::ValuesIn(epsMode),
            testing::ValuesIn(shapes),
            testing::ValuesIn(netPrecisions),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_NormalizeL2,
            KmbNormalizeL2LayerTest,
            normL2params,
            KmbNormalizeL2LayerTest::getTestCaseName
    );
}  // namespace
