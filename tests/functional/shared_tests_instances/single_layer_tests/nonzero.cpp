//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nonzero.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXNonZeroLayerTest_VPU3700 : public NonZeroLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
        }
    }
    void SkipBeforeValidate() override {
        throw LayerTestsUtils::KmbSkipTestException("comparison fails");
    }
};

TEST_P(VPUXNonZeroLayerTest_VPU3700, HW) {
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

// [Track number: S#43181]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_nonzero, VPUXNonZeroLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(additional_config)),
                         VPUXNonZeroLayerTest_VPU3700::getTestCaseName);

}  // namespace
