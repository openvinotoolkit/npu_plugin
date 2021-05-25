// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
namespace {
std::set<ngraph::helpers::EltwiseTypes> supportedTypesMCM {
    ngraph::helpers::EltwiseTypes::ADD,
    ngraph::helpers::EltwiseTypes::MULTIPLY,
    ngraph::helpers::EltwiseTypes::SUBTRACT,
    ngraph::helpers::EltwiseTypes::SQUARED_DIFF
};

std::set<ngraph::helpers::EltwiseTypes> supportedTypesMLIR {
    ngraph::helpers::EltwiseTypes::ADD,
    ngraph::helpers::EltwiseTypes::MULTIPLY,
    ngraph::helpers::EltwiseTypes::DIVIDE,
    ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
    ngraph::helpers::EltwiseTypes::POWER,
    ngraph::helpers::EltwiseTypes::FLOOR_MOD
};
} // namespace

class KmbEltwiseLayerTest: public EltwiseLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        ngraph::helpers::EltwiseTypes eltwiseOp;
        std::vector<std::vector<size_t>> inShapes;
        CommonTestUtils::OpType opType;
        std::set<std::vector<std::vector<size_t>>> scalarOnlyShapes = {
            {{1, 10, 100}},
            {{4, 4, 16}},
            {{1, 1, 1, 3}},
            {{1, 2, 4}},
            {{1, 4, 4}},
            {{1, 4, 4, 1}}
        };
        std::set<std::vector<std::vector<size_t>>> badShapesForMLIR = {
                {{2, 17, 5, 4}, {1, 17, 1, 1}},
                {{4, 4, 16}},
                {{1, 2, 4}},
                {{1, 4, 4}},
        };

        std::tie(inShapes,
                 eltwiseOp, std::ignore, opType, std::ignore,
                 std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();

        if (isCompilerMCM()) {
            if (supportedTypesMCM.find(eltwiseOp) ==
                supportedTypesMCM.end()) {
                throw LayerTestsUtils::KmbSkipTestException("Unsupported eltwise type in MCM compiler");
            }

            // Skip below is due to error during run of tests on KMB-board (it is oly for VECTOR OpType):
            // [Debug  ][VPU][VpualCoreNNExecutor] Allocated buffer for input with the size:
            // [Info   ][VPU][VpualCoreNNExecutor] allocateGraph begins
            // [Error  ][VPU][VpualCoreNNExecutor] allocateGraph: failed to create NnCorePlg
            // kmb-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:152: Failure
            // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
            // doesn't throw an exception.
            // Actual: it throws:VpualCoreNNExecutor::allocateGraph: failed to create NnCorePlg: 6
            // [Track number: S#51349]
            if (scalarOnlyShapes.find(inShapes) == scalarOnlyShapes.end() ||
                                                                opType == CommonTestUtils::OpType::VECTOR) {
                throw LayerTestsUtils::KmbSkipTestException("VECTOR OpType is unsupported");
            }
        } else {
            if (supportedTypesMLIR.find(eltwiseOp) ==
                supportedTypesMLIR.end()) {
                throw LayerTestsUtils::KmbSkipTestException("Experimental compiler doesn't supports this eltwise operation yet");
            }
            // Skip below is due to error during run of tests on KMB-board (it is for MLIR compiler only):
            // [Debug  ][VPU][VpualCoreNNExecutor] Allocated buffer for input with the size:
            // [Info   ][VPU][VpualCoreNNExecutor] allocateGraph begins
            // [Error  ][VPU][VpualCoreNNExecutor] allocateGraph: failed to create NnCorePlg
            // kmb-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:152: Failure
            // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
            // doesn't throw an exception.
            // Actual: it throws:VpualCoreNNExecutor::allocateGraph: failed to create NnCorePlg: 6
            // [Track number: S#51349]
            if ( badShapesForMLIR.find(inShapes) != badShapesForMLIR.end()) {
                throw LayerTestsUtils::KmbSkipTestException("Error on KMB-board: failed to create NnCorePlg: 6");
            }
            // Skip below is due to error during run of tests on KMB-board (MLIR compiler, 16 shaves used):
            // [Error  ][VPU][VpualCoreNNExecutor] pull: WaitForResponse failed
            // unknown file: Failure
            // C++ exception with description "VpualCoreNNExecutor::pull: WaitForResponse failed8" thrown in the test body.
            //        NnXlinkPlg: Close channel failed: 8
            // [Track number: S#11028]
            if (inShapes == std::vector<std::vector<size_t>>{{1, 4, 1, 1}} && opType == CommonTestUtils::OpType::VECTOR)  {
                throw LayerTestsUtils::KmbSkipTestException("Error on KMB-board: NnXlinkPlg: Close channel failed: 8");
            }
        }
    }

    // There are errors at validation step on KMB-board for some input shapes:
    // KmbLayerTestsCommon::Validate()
    // LayerTestsCommon::Validate()
    // openvino/inference-engine/tests/functional/shared_test_classes/include/
    // shared_test_classes/base/layer_test_utils.hpp:173: Failure
    // Value of: max != 0 && (diff <= static_cast<float>(threshold))
    // Actual: false
    // Expected: true
    // Relative comparison of values expected: -4 and actual: 0 at index 1 with
    // threshold 0.0099999997764825821 failed
    // [Track number: S#51346]
    void SkipBeforeValidate() override {
        ngraph::helpers::EltwiseTypes eltwiseOp;
        std::vector<std::vector<size_t>> inShapes;
        std::tie(inShapes,
                 eltwiseOp, std::ignore, std::ignore, std::ignore,
                 std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();
        std::set<std::vector<std::vector<size_t>>> badShapes = {
            {{2, 17, 5, 4}, {1, 17, 1, 1}},
            {{1, 4, 4, 1}}
        };
        if ( badShapes.find(inShapes) != badShapes.end()  ) {
            throw LayerTestsUtils::KmbSkipTestException("Mismatch in comparison");
        }
    }

    void SetUp() override {
        EltwiseLayerTest::SetUp();
    }
};

TEST_P(KmbEltwiseLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbEltwiseLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
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
    {{2, 17, 5, 4}, {1, 17, 1, 1}},
    {{2, 17, 5, 1}, {1, 17, 1, 4}},
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
    ngraph::helpers::EltwiseTypes::ADD,
    ngraph::helpers::EltwiseTypes::MULTIPLY,
    ngraph::helpers::EltwiseTypes::SUBTRACT,
    ngraph::helpers::EltwiseTypes::DIVIDE,
    ngraph::helpers::EltwiseTypes::FLOOR_MOD,
    ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
    ngraph::helpers::EltwiseTypes::POWER,
    ngraph::helpers::EltwiseTypes::MOD
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
    {{1, 4, 4, 1}},
    {{1, 4, 1, 1}},
    {{1, 10, 100}},
    {{4, 4, 16}},
    {{1, 1, 1, 3}},
    {{2, 17, 5, 4}, {1, 17, 1, 1}},
    {{1, 2, 4}},
    {{1, 4, 4}}
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes_pass_mcm = {
    ngraph::helpers::InputLayerType::CONSTANT,
};

std::vector<CommonTestUtils::OpType> opTypes_pass_mcm = {
    CommonTestUtils::OpType::SCALAR,
    CommonTestUtils::OpType::VECTOR,
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
