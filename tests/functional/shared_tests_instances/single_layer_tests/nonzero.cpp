//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/nonzero.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class NonZeroLayerTest_NPU3700 : public NonZeroLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SkipBeforeLoad() override {
        if (envConfig.IE_NPU_TESTS_RUN_INFER) {
            throw LayerTestsUtils::VpuSkipTestException("layer test networks hang the board");
        }
    }
    void SkipBeforeValidate() override {
        throw LayerTestsUtils::VpuSkipTestException("comparison fails");
    }
};

TEST_P(NonZeroLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
std::vector<std::vector<size_t>> inShapes = {
        {1000}, {4, 1000}, {2, 4, 1000}, {2, 4, 4, 1000}, {2, 4, 4, 2, 1000},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::U8,
};

std::map<std::string, std::string> additional_config = {};

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_nonzero, NonZeroLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(additional_config)),
                         NonZeroLayerTest_NPU3700::getTestCaseName);

}  // namespace
