// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/convert.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConvertLayerTest: public ConvertLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I8,
};

// Tests fail with one of two common errors:
// 1. C++ exception with description "Unsupported operation: Convert_7756 with name Convert_7843 with
// type Convert with C++ type N6ngraph2op2v07ConvertE
// kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1431
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
//
// 2. C++ exception with description "Input image format I8 is not supported yet.
// Supported formats:F16, FP32 and U8.
// kmb-plugin/src/kmb_plugin/kmb_plugin.cpp:53
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#41523]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_NoReshape, KmbConvertLayerTest,
                        ::testing::Combine(
                            ::testing::Values(inShape),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvertLayerTest::getTestCaseName);

}  // namespace
