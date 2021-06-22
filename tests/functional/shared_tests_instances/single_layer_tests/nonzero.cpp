// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/nonzero.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
 
namespace LayerTestsDefinitions {

    class KmbNonZeroLayerTest: public NonZeroLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeLoad() override {
            if (envConfig.IE_KMB_TESTS_RUN_INFER) {
                throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
            }
        }
        void SkipBeforeValidate() override {
            throw LayerTestsUtils::KmbSkipTestException("comparison fails");
        }
    };

    TEST_P(KmbNonZeroLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {
    std::vector<std::vector<size_t>> inShapes = {
            {1000},
            {4, 1000},
            {2, 4, 1000},
            {2, 4, 4, 1000},
            {2, 4, 4, 2, 1000},
    };

    const std::vector<InferenceEngine::Precision> inputPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::U8,
    };

    std::map<std::string, std::string> additional_config = {};

    // Tests fails with one of common errors:
    // 1. C++ exception with description "Unsupported operation: NonZero_3972 with name NonZero_3987 with
    // type NonZero with C++ type N6ngraph2op2v37NonZeroE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1524
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 2. C++ exception with description "Unsupported dimensions layout
    // kmb-plugin/src/utils/dims_parser.cpp:45
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // [Track number: S#43181]
    INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_nonzero, KmbNonZeroLayerTest,
            ::testing::Combine(
                    ::testing::ValuesIn(inShapes),
                    ::testing::ValuesIn(inputPrecisions),
                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                    ::testing::Values(additional_config)),
            KmbNonZeroLayerTest::getTestCaseName);

}  // namespace
