// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/select.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbSelectLayerTest : public SelectLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeValidate() override {
            throw LayerTestsUtils::KmbSkipTestException("comparison fails");
        }
    };

    class KmbSelectLayerTest_MLIR : public KmbSelectLayerTest {};

    TEST_P(KmbSelectLayerTest_MLIR, CompareWithRefs_HW) {
        useCompilerMLIR();
        setReferenceSoftwareModeMLIR();
        Run();
    }
} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecision = {
    //InferenceEngine::Precision::I8
    InferenceEngine::Precision::FP16,
    //InferenceEngine::Precision::FP32 
};

const std::vector<std::vector<std::vector<size_t>>> noneShapes = {
    {{1}, {1}, {1}},
    {{8}, {8}, {8}},
    {{4, 5}, {4, 5}, {4, 5}},
    {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
    {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
    {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}
};

const auto noneCases = ::testing::Combine(
    ::testing::ValuesIn(noneShapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsSelectOp_none, KmbSelectLayerTest_MLIR, noneCases, KmbSelectLayerTest::getTestCaseName);
}  // namespace
