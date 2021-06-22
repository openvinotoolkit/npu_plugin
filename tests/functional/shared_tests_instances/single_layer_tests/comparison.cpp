// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/comparison.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbComparisonLayerTest : public ComparisonLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeLoad() override {
            if (envConfig.IE_KMB_TESTS_RUN_INFER) {
                throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
            }
        }
        void SkipBeforeValidate() override {
            throw LayerTestsUtils::KmbSkipTestException("comparison fails");
        }
    };

    TEST_P(KmbComparisonLayerTest, CompareWithRefs) {
        Run();
    }
} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {

    std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
            {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
            {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
            {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
            {{1, 3, 20}, {{20}, {2, 1, 1}}},
            {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
            {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
    };

    std::vector<InferenceEngine::Precision> inputsPrecisions = {
            InferenceEngine::Precision::FP32,
    };

    std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes = {
            ngraph::helpers::ComparisonTypes::EQUAL,
            ngraph::helpers::ComparisonTypes::NOT_EQUAL,
            ngraph::helpers::ComparisonTypes::GREATER,
            ngraph::helpers::ComparisonTypes::GREATER_EQUAL,
            ngraph::helpers::ComparisonTypes::LESS,
            ngraph::helpers::ComparisonTypes::LESS_EQUAL,
    };

    std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
            ngraph::helpers::InputLayerType::CONSTANT,
            ngraph::helpers::InputLayerType::PARAMETER,
    };

    std::map<std::string, std::string> additional_config = {};

    const auto ComparisonTestParams = ::testing::Combine(
            ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
            ::testing::ValuesIn(inputsPrecisions),
            ::testing::ValuesIn(comparisonOpTypes),
            ::testing::ValuesIn(secondInputTypes),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
            ::testing::Values(additional_config));

    // There are 312 instances of this test. All of them fail with one of common errors:
    // 1. C++ exception with description "Unsupported operation: LessEqual_204951 with name LessEqual_205020
    // with type LessEqual with C++ type N6ngraph2op2v19LessEqualE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1518
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 2. C++ exception with description "Unsupported operation: Less_203633 with name Less_203702 with
    // type Less with C++ type N6ngraph2op2v14LessE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1518
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 3. C++ exception with description "Unsupported operation: GreaterEqual_202315 with name GreaterEqual_202384
    // with type GreaterEqual with C++ type N6ngraph2op2v112GreaterEqualE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1518
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 4. C++ exception with description "Unsupported operation: Greater_200997 with name Greater_201066 with
    // type Greater with C++ type N6ngraph2op2v17GreaterE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1518
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 5. C++ exception with description "Unsupported operation: NotEqual_199679 with name NotEqual_199748 with
    // type NotEqual with C++ type N6ngraph2op2v18NotEqualE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1518
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 6. C++ exception with description "Unsupported operation: Equal_198361 with name Equal_198430 with
    // type Equal with C++ type N6ngraph2op2v15EqualE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1518
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // 7. C++ exception with description "Unsupported dimensions layout
    // kmb-plugin/src/utils/dims_parser.cpp:45
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
    //
    // [Track number: S#43012]
    INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_CompareWithRefs, KmbComparisonLayerTest, ComparisonTestParams, KmbComparisonLayerTest::getTestCaseName);

}  // namespace
