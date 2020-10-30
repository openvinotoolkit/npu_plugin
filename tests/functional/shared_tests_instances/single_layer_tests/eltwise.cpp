// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbEltwiseLayerTest: public EltwiseLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeImport() override {
        throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
    }
};

TEST_P(KmbEltwiseLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{2}},
    {{2, 200}},
    {{10, 200}},
    {{1, 10, 100}},
    {{4, 4, 16}},
    {{1, 1, 1, 3}},
    {{1, 2, 4}},
    {{1, 4, 4}},
    {{1, 4, 4, 1}},
    {{1, 1, 1, 1, 1, 1, 3}},
    {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
    CommonTestUtils::OpType::SCALAR,
    CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {
    ngraph::helpers::EltwiseTypes::MULTIPLY,
    ngraph::helpers::EltwiseTypes::SUBTRACT,
    ngraph::helpers::EltwiseTypes::ADD
};

std::map<std::string, std::string> additional_config = {};

const auto multiply_params = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(eltwiseOpTypes),
    ::testing::ValuesIn(secondaryInputTypes),
    ::testing::ValuesIn(opTypes),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    ::testing::Values(additional_config));

// Test works only when input/outpu layouts are NHWC. There are two such tests:
// 1) DISABLED_CompareWithRefs/KmbEltwiseLayerTest.CompareWithRefs/IS=(1.1.1.3)_eltwiseOpType=
//                                          Prod_secondaryInputType=CONSTANT_opType=SCALAR_netPRC=FP32_targetDevice=KMB
// 2) DISABLED_CompareWithRefs/KmbEltwiseLayerTest.CompareWithRefs/IS=(1.1.1.3)_eltwiseOpType=
//                                          Prod_secondaryInputType=CONSTANT_opType=SCALAR_netPRC=FP16_targetDevice=KMB
// In other cases it shows errors like these:
// C++ exception with description "Size of dims(1) and format(NHWC) are inconsistent.
// C++ exception with description "Size of dims(2) and format(NHWC) are inconsistent.
// C++ exception with description "Size of dims(3) and format(NHWC) are inconsistent.
// There is segmentation fault for
// CompareWithRefs/KmbEltwiseLayerTest.CompareWithRefs/IS=(1.1.1.3)_eltwiseOpType=
//                                          Prod_secondaryInputType=CONSTANT_opType=VECTOR_netPRC=FP32_targetDevice=KMB
// [Track number: S#39979]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_CompareWithRefs, KmbEltwiseLayerTest, multiply_params,
                        KmbEltwiseLayerTest::getTestCaseName);

// Below is subset of parameters and additional test based on DISABLED_smoke_CompareWithRefs (see above).
// It is created just to enable running part of initial test for eltwise layer.
// Do not forget to remove it and corresponding variables when main test DISABLED_smoke_CompareWithRefs
// will work properly.
std::vector<std::vector<std::vector<size_t>>> inShapes_pass_mcm = {
    {{1, 1, 1, 3}},
    {{1, 4, 4, 1}},
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes_pass_mcm = {
    ngraph::helpers::InputLayerType::CONSTANT,
};

std::vector<CommonTestUtils::OpType> opTypes_pass_mcm = {
    CommonTestUtils::OpType::SCALAR,
};

const auto multiply_params_pass_mcm = ::testing::Combine(
    ::testing::ValuesIn(inShapes_pass_mcm),
    ::testing::ValuesIn(eltwiseOpTypes),
    ::testing::ValuesIn(secondaryInputTypes_pass_mcm),
    ::testing::ValuesIn(opTypes_pass_mcm),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs_pass_mcm, KmbEltwiseLayerTest, multiply_params_pass_mcm,
                        KmbEltwiseLayerTest::getTestCaseName);
// End of additional test and its parameters

}  // namespace
