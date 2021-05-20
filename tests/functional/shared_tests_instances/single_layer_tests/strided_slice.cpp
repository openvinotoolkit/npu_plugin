// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/strided_slice.hpp"

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbStridedSliceLayerTest : public StridedSliceLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbStridedSliceLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<StridedSliceSpecificParams> ss_only_test_cases = {
        {{2, 2, 2, 2}, {0, 0, 0, 0}, {2, 2, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        {{2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, {}, {}, {}},
        {{2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {}, {}, {}},
        {{2, 3, 2, 4}, {0, 0, 0, 0}, {2, 2, 4, 3}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},

        {{1, 3, 2, 4}, {0, 0, 0, 0}, {1, 3, 2, 4}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        {{1, 2, 3, 4}, {0, 0, 0, 0}, {1, 2, 3, 4}, {1, 1, 1, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        {{1, 3, 4, 2}, {0, 0, 0, 0}, {1, 3, 4, 2}, {1, 1, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        {{1, 2, 3, 4}, {0, 0, 0, 0}, {1, 2, 3, 4}, {1, 1, 1, 2}, {0, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        {{1, 1, 2, 3}, {0, 0, 0, 0}, {1, 1, 2, 3}, {1, 1, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},

        {{2, 2, 4, 2}, {0, 0, 0, 0}, {2, 2, 4, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        {{2, 2, 4, 2}, {1, 0, 0, 1}, {2, 2, 4, 2}, {1, 1, 2, 1}, {0, 1, 1, 0}, {1, 1, 0, 0}, {}, {}, {}},
};

using Config = std::map<std::string, std::string>;

Config getConfig() {
    return Config{{"VPU_COMPILER_ALLOW_NCHW_MCM_INPUT", "YES"}};
}

INSTANTIATE_TEST_CASE_P(smoke_StridedSlice0, KmbStridedSliceLayerTest,
                       ::testing::Combine(::testing::ValuesIn(ss_only_test_cases),
                                          ::testing::Values(InferenceEngine::Precision::FP16),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          ::testing::Values(InferenceEngine::Layout::NCHW),
                                          ::testing::Values(InferenceEngine::Layout::NCHW),
                                          ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                          ::testing::Values(getConfig())),
                       StridedSliceLayerTest::getTestCaseName);

 INSTANTIATE_TEST_CASE_P(smoke_StridedSlice1, KmbStridedSliceLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ss_only_test_cases),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(getConfig())),
                         StridedSliceLayerTest::getTestCaseName);

}  // namespace
