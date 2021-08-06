// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/eltwise.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbEltwiseLayerTest:
        public EltwiseLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

class KmbEltwiseLayerTest_MCM : public KmbEltwiseLayerTest {
    void SkipBeforeValidate() override {
        std::vector<std::vector<size_t>> inShapes;
        std::tie(inShapes,
                 std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();

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

        std::set<std::vector<std::vector<size_t>>> badShapes = {
                {{2, 200}},
                {{10, 200}},
                {{1, 4, 4, 1}},
                {{2, 17, 5, 4}, {1, 17, 1, 1}},
                {{2, 17, 5, 1}, {1, 17, 1, 4}}
        };

        if (badShapes.count(inShapes)) {
            throw LayerTestsUtils::KmbSkipTestException("Mismatch in comparison");
        }
    }
};
class KmbEltwiseLayerTest_MLIR : public KmbEltwiseLayerTest {
    void SkipBeforeLoad() override {
        ngraph::helpers::EltwiseTypes eltwiseOp;
        std::vector<std::vector<size_t>> inShapes;
        ngraph::helpers::InputLayerType secondInputType;
        CommonTestUtils::OpType opType;

        std::tie(inShapes,
                 eltwiseOp, secondInputType, opType, std::ignore,
                 std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();


        std::set<ngraph::helpers::EltwiseTypes> fusedToScaleShiftOpMLIR = {
                ngraph::helpers::EltwiseTypes::ADD,
                ngraph::helpers::EltwiseTypes::MULTIPLY
        };

        // A special workaround([Track number: E#13127]) extends all network inputs/outputs to 4D,
        // which causes an incorrect conversion of the eltwise operation to a ScaleShift.
        // [Track number: E#13311]
        if (fusedToScaleShiftOpMLIR.count(eltwiseOp) && secondInputType == ngraph::helpers::InputLayerType::PARAMETER)  {
            throw LayerTestsUtils::KmbSkipTestException("Skipping the operation due to incorrect conversion to ScaleShift");
        }

        std::set<std::vector<std::vector<size_t>>> shapesWithBatch = {
                {{2, 17, 5, 1}, {1, 17, 1, 4}},
        };
        std::set<std::vector<std::vector<size_t>>> fusedToScaleShiftShapes = {
                {{2, 17, 5, 4}, {1, 17, 1, 1}},
        };

        // A special workaround([Track number: E#13127]) extends all network inputs/outputs to 4D,
        // which causes error: "Batch size != 1 is not supported"
        // [Track number: E#7613]
        if ((fusedToScaleShiftShapes.count(inShapes) && fusedToScaleShiftOpMLIR.count(eltwiseOp))
            || shapesWithBatch.count(inShapes) )  {
            throw LayerTestsUtils::KmbSkipTestException("Skipping the operation due to incorrect conversion to ScaleShift");
        }
    }

    void SkipBeforeValidate() override {
        ngraph::helpers::EltwiseTypes eltwiseOp;
        std::vector<std::vector<size_t>> inShapes;
        CommonTestUtils::OpType opType;
        std::tie(inShapes,
                 eltwiseOp, std::ignore, opType, std::ignore,
                 std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) = GetParam();

        std::set<ngraph::helpers::EltwiseTypes> fusedToScaleShiftOpMLIR = {
                ngraph::helpers::EltwiseTypes::ADD,
        };

        std::set<std::vector<std::vector<size_t>>> fusedToScaleShiftOpShapes = {
                {{1, 4, 1, 1}},
                {{1, 4, 4, 1}},
        };

        // A special workaround([Track number: E#13127]) extends all network inputs/outputs to 4D,
        // which causes an incorrect conversion of the eltwise operation to a ScaleShift.
        // At the same time, per-channel broadcast doesn't work correctly in ScaleShift layer
        // which leads to accuracy errors
        // [Track number: E#13311]
        if (fusedToScaleShiftOpMLIR.count(eltwiseOp) && fusedToScaleShiftOpShapes.count(inShapes))  {
            throw LayerTestsUtils::KmbSkipTestException("Skipping the operation due to incorrect conversion to ScaleShift");
        }

        std::set<std::vector<std::vector<size_t>>> vectorOnlyShapesMLIR = {
                {{4, 4, 16}},
        };

        // There are errors at validation step on KMB-board for some input shapes:
        // [Track number: S#51346]
        if (vectorOnlyShapesMLIR.count(inShapes) && opType == CommonTestUtils::OpType::SCALAR) {
            throw LayerTestsUtils::KmbSkipTestException("Mismatch in comparison");
        }

        std::set<ngraph::helpers::EltwiseTypes> badOpMLIR = {
                ngraph::helpers::EltwiseTypes::DIVIDE,
                ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
                ngraph::helpers::EltwiseTypes::POWER,
                ngraph::helpers::EltwiseTypes::FLOOR_MOD
        };

        std::set<std::vector<std::vector<size_t>>> badShapesMLIR = {
                {{2}},
                {{2, 200}},
                {{10, 200}},
                {{2, 17, 5, 4}, {1, 17, 1, 1}},
        };

        // There are errors at validation step on KMB-board for some input shapes:
        // [Track number: S#51346]
        if (badOpMLIR.count(eltwiseOp) && badShapesMLIR.count(inShapes)) {
            throw LayerTestsUtils::KmbSkipTestException("Mismatch in comparison");
        }
    }
};


//
//[Track number: E#15146]
//
TEST_P(KmbEltwiseLayerTest_MCM, DISABLED_CompareWithRefs) {
    Run();
}

// [Track number: E#15146]
// Initialization disabled partly
TEST_P(KmbEltwiseLayerTest_MLIR, CompareWithRefs) {
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
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{4, 4, 16}},
        {{1, 10, 100}},
        {{1, 4, 1, 1}},
        {{1, 1, 1, 3}},
        {{1, 4, 4, 1}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::PARAMETER,
    ngraph::helpers::InputLayerType::CONSTANT,
};

std::vector<CommonTestUtils::OpType> opTypes = {
    CommonTestUtils::OpType::VECTOR,
    CommonTestUtils::OpType::SCALAR,
};

std::map<std::string, std::string> additional_config = {};

//
// MCM Instantiation
//

std::set<ngraph::helpers::EltwiseTypes> supportedTypesMCM {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF
};

const auto eltwise_params_mcm = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(supportedTypesMCM),
    ::testing::ValuesIn(secondaryInputTypes),
    ::testing::Values(CommonTestUtils::OpType::SCALAR),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    ::testing::Values(InferenceEngine::Layout::ANY),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, KmbEltwiseLayerTest_MCM, eltwise_params_mcm,
                        KmbEltwiseLayerTest::getTestCaseName);

//
//[Track number: S#51349]
//

// Skip below is due to error during run of tests on KMB-board (it is oly for VECTOR OpType):
// [Debug  ][VPU][VpualCoreNNExecutor] Allocated buffer for input with the size:
// [Info   ][VPU][VpualCoreNNExecutor] allocateGraph begins
// [Error  ][VPU][VpualCoreNNExecutor] allocateGraph: failed to create NnCorePlg
// kmb-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:152: Failure
// Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration)
// doesn't throw an exception.
// Actual: it throws:VpualCoreNNExecutor::allocateGraph: failed to create NnCorePlg: 6

const auto eltwise_params_vector_mcm = ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(supportedTypesMCM),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::Values(CommonTestUtils::OpType::VECTOR),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_CompareWithRefs, KmbEltwiseLayerTest_MCM, eltwise_params_vector_mcm,
                        KmbEltwiseLayerTest::getTestCaseName);

//
// MLIR Instantiation
//

std::set<ngraph::helpers::EltwiseTypes> supportedTypesMLIR {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::DIVIDE,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        ngraph::helpers::EltwiseTypes::POWER,
        ngraph::helpers::EltwiseTypes::FLOOR_MOD
};

const auto eltwise_params_mlir = ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(supportedTypesMLIR),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config));

// [Track number: E#15146]
// Initialization disabled partly
/*
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, KmbEltwiseLayerTest_MLIR, eltwise_params_mlir,
                        KmbEltwiseLayerTest::getTestCaseName);
*/

// Specific multiply case

std::vector<std::vector<std::vector<size_t>>> inSpecificMultiplyShapes = {
        {{1, 3, 224, 224}, {1, 1, 1, 1}},  
};

const auto multiply_params_mlir = ::testing::Combine(
        ::testing::ValuesIn(inSpecificMultiplyShapes),
        ::testing::Values(ngraph::helpers::EltwiseTypes::MULTIPLY),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_Specific, KmbEltwiseLayerTest_MLIR, multiply_params_mlir,
                        KmbEltwiseLayerTest::getTestCaseName);

}  // namespace
