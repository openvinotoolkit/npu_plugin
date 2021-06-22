// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reduce_ops.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
    class KmbReduceOpsLayerTest : public ReduceOpsLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeValidate() override {
            const auto testName =
                    std::string{::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()};

            const auto skipMCM = testName.find("SKIP_MCM") != std::string::npos;
            if (isCompilerMCM() && skipMCM) {
                throw LayerTestsUtils::KmbSkipTestException("Skip validate for MCM");
            }
        }
    };

    TEST_P(KmbReduceOpsLayerTest, CompareWithRefs) {
        Run();
    }

   TEST_P(KmbReduceOpsLayerTest, CompareWithRefs_MLIR) {
       useCompilerMLIR();
       Run();
   }

    class KmbReduceOpsLayerWithSpecificInputTest : public ReduceOpsLayerWithSpecificInputTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeLoad() override {
            if (isCompilerMCM()) {
                throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
            }
        }
        void SkipBeforeValidate() override {
            if (isCompilerMCM()) {
                throw LayerTestsUtils::KmbSkipTestException("comparison fails");
            }
        }
    };

    TEST_P(KmbReduceOpsLayerWithSpecificInputTest, CompareWithRefs) {
        Run();
    }

    TEST_P(KmbReduceOpsLayerWithSpecificInputTest, CompareWithRefs_MLIR) {
        useCompilerMLIR();
        Run();
    }

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16, // CPU-plugin has parameter I32, but KMB does not
            InferenceEngine::Precision::U8    // support it. So I32 is changed to FP16 and U8.
    };

    const std::vector<bool> keepDims = {
            true,
            false,
    };

    const std::vector<std::vector<size_t>> inputShapes = {
            std::vector<size_t>{10, 20, 30, 40},
            std::vector<size_t>{3, 5, 7, 9},
    };

    const std::vector<std::vector<size_t>> inputShapesOneAxis = {
            std::vector<size_t>{10, 20, 30, 40},
            std::vector<size_t>{3, 5, 7, 9},
            std::vector<size_t>{10},
    };

    const std::vector<std::vector<int>> axes = {
            {0},
            {1},
            {2},
            {3},
            {0, 1},
            {0, 2},
            {0, 3},
            {1, 2},
            {1, 3},
            {2, 3},
            {0, 1, 2},
            {0, 1, 3},
            {0, 2, 3},
            {1, 2, 3},
            {0, 1, 2, 3},
            {1, -1}
    };

    std::vector<CommonTestUtils::OpType> opTypes = {
            CommonTestUtils::OpType::SCALAR,
            CommonTestUtils::OpType::VECTOR,
    };

    const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
            ngraph::helpers::ReductionType::Mean,
            ngraph::helpers::ReductionType::Min,
            ngraph::helpers::ReductionType::Max,
            ngraph::helpers::ReductionType::Sum,
            ngraph::helpers::ReductionType::Prod,
            ngraph::helpers::ReductionType::LogicalOr,
            ngraph::helpers::ReductionType::LogicalAnd,
    };

    // [Track number: S#43428]
    INSTANTIATE_TEST_SUITE_P(
            DISABLED_smoke_ReduceOneAxis,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::Values(std::vector<int>{0}),
                testing::ValuesIn(opTypes),
                testing::Values(true, false),
                testing::ValuesIn(reductionTypes),
                testing::Values(InferenceEngine::Precision::FP32),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::ValuesIn(inputShapesOneAxis),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)
            ),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // [Track number: S#43428]
    // ReduceSum is not supported, replacement pass does not match.
    INSTANTIATE_TEST_SUITE_P(
            DISABLED_smoke_Reduce_Precisions,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::Values(std::vector<int>{1, 3}),
                testing::Values(opTypes[1]),
                testing::ValuesIn(keepDims),
                testing::Values(ngraph::helpers::ReductionType::Sum),
                testing::Values(InferenceEngine::Precision::FP32,
                                InferenceEngine::Precision::FP16, // CPU-plugin has parameter I32, but KMB does not
                                InferenceEngine::Precision::U8), // support it. So I32 is changed to FP16 and U8.
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
        smoke_ReduceSum,
        KmbReduceOpsLayerWithSpecificInputTest,
        testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}}),
                testing::Values(opTypes[1]),
                testing::Values(true),
                testing::Values(ngraph::helpers::ReductionType::Sum),
                testing::Values(InferenceEngine::Precision::FP32),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t> {1, 512, 7, 7}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbReduceOpsLayerWithSpecificInputTest::getTestCaseName
    );

    // Test hangs on x86 when executes test-case
    // smoke_Reduce_Axes/KmbReduceOpsLayerTest.CompareWithRefs/IS=(10.20.30.40)_axes=(1)_opType=VECTOR_
    // type=Mean_KeepDims_netPRC=FP32_inPRC=UNSPECIFIED_outPRC=UNSPECIFIED_inL=ANY_trgDev=KMB
    // [Track number: S#43428]
    INSTANTIATE_TEST_SUITE_P(
            DISABLED_smoke_Reduce_Axes,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::ValuesIn(axes),
                testing::Values(opTypes[1]),
                testing::ValuesIn(keepDims),
                testing::Values(ngraph::helpers::ReductionType::Mean),
                testing::Values(InferenceEngine::Precision::FP32),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::ValuesIn(inputShapes),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)
            ),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // {2, 4, 6, 8} case is replaced with Reshape to {4, 6, 8}, that MCM fails to compile
    // Tensor:Parameter_375:0 - ArgumentError: attribute identifer allocators - Undefined identifier
    INSTANTIATE_TEST_SUITE_P(
            smoke_ReduceOneAxis_SKIP_MCM,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}, {1}}),
                testing::Values(opTypes[1]),
                testing::Values(true, false),
                testing::Values(ngraph::helpers::ReductionType::Mean,
                                ngraph::helpers::ReductionType::Max),
                testing::Values(InferenceEngine::Precision::FP32),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{3, 5},
                                std::vector<size_t>{2, 4, 6},
                                std::vector<size_t>{2, 4, 6, 8}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_Reduce_from_networks,
            KmbReduceOpsLayerTest,
            testing::Combine(
                    testing::ValuesIn(decltype(axes) { {2, 3} }),
                    testing::Values(opTypes[1]),
                    testing::Values(true, false),
                    testing::Values(ngraph::helpers::ReductionType::Mean),
                    testing::Values(InferenceEngine::Precision::FP32),
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Layout::ANY),
                    testing::Values(
                        std::vector<size_t> {1, 512, 7, 7},     // resnet_18
                        std::vector<size_t> {1, 2048, 7, 7},    // resnet_50
                        std::vector<size_t> {1, 1280, 7, 7},    // mobilenet_v2
                        std::vector<size_t> {1, 1664, 7, 7}     // densenet
                    ),
                    testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
            DISABLED_smoke_Reduce_from_networks,
            KmbReduceOpsLayerTest,
            testing::Combine(
                    testing::ValuesIn(decltype(axes) { {2, 3} }),
                    testing::Values(opTypes[1]),
                    testing::Values(true, false),
                    testing::Values(ngraph::helpers::ReductionType::Mean),
                    testing::Values(InferenceEngine::Precision::FP32),
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                    testing::Values(InferenceEngine::Layout::ANY),
                    testing::Values(
                        // [Track number: S#43428]
                        std::vector<size_t> {1, 1000, 14, 14},  // squeezenet1_1 (the network is working)
                        // ERROR:   LpScheduler - RuntimeError: Both start time and completion heaps are empty.
                        std::vector<size_t> {100, 91, 6, 6},    // rfcn_resnet101
                        std::vector<size_t> {100, 360, 6, 6},   // rfcn_resnet101
                        std::vector<size_t> {100, 2048, 7, 7}   // faster_rcnn_resnet101
                    ),
                    testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerTest::getTestCaseName
    );

}  // namespace
