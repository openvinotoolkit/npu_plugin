//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reduce_ops.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include <common/functions.h>

namespace LayerTestsDefinitions {
    class KmbReduceOpsLayerTest : public ReduceOpsLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SetUp() override {
            InferenceEngine::Precision netPrecision;
            CommonTestUtils::OpType opType;
            ngraph::helpers::ReductionType reductionType;
            ngraph::NodeVector convertedInputs;
            ngraph::OutputVector paramOuts;
            std::shared_ptr<ngraph::Node> reduceNode;
            std::vector<size_t> inputShape, shapeAxes;
            std::vector<int> axes;
            bool keepDims;

            std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inLayout, inputShape,
                    targetDevice) = GetParam();

            if ((reductionType == ngraph::helpers::ReductionType::LogicalOr)||(reductionType == ngraph::helpers::ReductionType::LogicalAnd)) {
                auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
                auto inputs = ngraph::builder::makeParams(ngPrc, {inputShape});

                switch (opType) {
                case CommonTestUtils::OpType::SCALAR: {
                    if (axes.size() > 1)
                        FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
                    break;
                }
                case CommonTestUtils::OpType::VECTOR: {
                    shapeAxes.push_back(axes.size());
                    break;
                }
                default:
                    FAIL() << "Reduce op doesn't support operation type: " << opType;
                }
                auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<ngraph::opset3::Constant>(
                        ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

                // Boolean type is unsupported in vpux-plugin, this is why the Convert layers are used to convert the input type into ngraph::element::boolean.
                for (const auto& input : inputs) {
                    convertedInputs.push_back(std::make_shared<ngraph::opset5::Convert>(input, ngraph::element::boolean));
                }
                paramOuts = ngraph::helpers::convert2OutputVector(convertedInputs);
                reduceNode = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);
                const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(reduceNode)};
                function = std::make_shared<ngraph::Function>(results, inputs, "Reduce");
            } else {
                ReduceOpsLayerTest::SetUp();
            }
        }
        void SkipBeforeLoad() override {
            const auto testName =
                    std::string{::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()};
            const auto skipMCM = testName.find("SKIP_MCM") != std::string::npos;
            if (isCompilerMCM() && skipMCM) {
                throw LayerTestsUtils::KmbSkipTestException("Skip load for MCM");
            }
        }
        void SkipBeforeValidate() override {
            const auto testName =
                    std::string{::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()};
            const auto skipMCM = testName.find("SKIP_MCM") != std::string::npos;
            if (isCompilerMCM() && skipMCM) {
                throw LayerTestsUtils::KmbSkipTestException("Skip validate for MCM");
            }
        }
        void SkipBeforeInfer() override {
            const auto testName =
                    std::string{::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()};
            const auto skipMCM = testName.find("smoke_ReduceOneAxis_SKIP_MCM") != std::string::npos;
            // [Track number: E#20269]
            if (getBackendName(*getCore()) == "LEVEL0" && skipMCM) {
                throw LayerTestsUtils::KmbSkipTestException("Level0 exception: dims and format are inconsistent.");
            }
        }
    };

    TEST_P(KmbReduceOpsLayerTest, CompareWithRefs) {
        Run();
    }

   TEST_P(KmbReduceOpsLayerTest, CompareWithRefs_MLIR) {
       useCompilerMLIR();
       setReferenceSoftwareModeMLIR();
       Run();
   }

   TEST_P(KmbReduceOpsLayerTest, CompareWithRefs_MLIR_HW) {
       useCompilerMLIR();
       setDefaultHardwareModeMLIR();
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
            InferenceEngine::Precision::FP16
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

    const std::vector<InferenceEngine::Layout> layouts3D = {
            InferenceEngine::Layout::CHW,
            InferenceEngine::Layout::HWC,
    };

    const std::vector<InferenceEngine::Layout> layouts4D = {
            InferenceEngine::Layout::NCHW,
            InferenceEngine::Layout::NHWC,
    };

    // [Track number: S#43428]
    INSTANTIATE_TEST_SUITE_P(
            DISABLED_smoke_ReduceOneAxis,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}}),
                testing::ValuesIn(opTypes),
                testing::Values(true, false),
                testing::ValuesIn(reductionTypes),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::ValuesIn(inputShapesOneAxis),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)
            ),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_Reduce_Precisions_SKIP_MCM,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{1, 3}}),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::Values(ngraph::helpers::ReductionType::Sum,
                                ngraph::helpers::ReductionType::Min,
                                ngraph::helpers::ReductionType::L1,
                                ngraph::helpers::ReductionType::LogicalOr,
                                ngraph::helpers::ReductionType::LogicalAnd,
                                ngraph::helpers::ReductionType::Prod,
                                ngraph::helpers::ReductionType::L2),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_Reduce,
            KmbReduceOpsLayerWithSpecificInputTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}}),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true, false),
                testing::Values(ngraph::helpers::ReductionType::Max,
                                ngraph::helpers::ReductionType::Sum,
                                ngraph::helpers::ReductionType::Min,
                                ngraph::helpers::ReductionType::L1,
                                ngraph::helpers::ReductionType::LogicalOr,
                                ngraph::helpers::ReductionType::LogicalAnd,
                                ngraph::helpers::ReductionType::Prod,
                                ngraph::helpers::ReductionType::L2),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t> {1, 512, 7, 7}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerWithSpecificInputTest::getTestCaseName
    );

    // [Track number: E#22733]
    INSTANTIATE_TEST_SUITE_P(
            DISABLED_smoke_Reduce3D,
            KmbReduceOpsLayerWithSpecificInputTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}}),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::Values(ngraph::helpers::ReductionType::Mean,
                                ngraph::helpers::ReductionType::Min,
                                ngraph::helpers::ReductionType::L1,
                                ngraph::helpers::ReductionType::LogicalOr,
                                ngraph::helpers::ReductionType::LogicalAnd,
                                ngraph::helpers::ReductionType::Prod,
                                ngraph::helpers::ReductionType::L2),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::ValuesIn(layouts3D),
                testing::Values(std::vector<size_t> {512, 7, 7}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerWithSpecificInputTest::getTestCaseName
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_Reduce4D,
            KmbReduceOpsLayerWithSpecificInputTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}}),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::Values(ngraph::helpers::ReductionType::Mean,
                                ngraph::helpers::ReductionType::Min,
                                ngraph::helpers::ReductionType::L1,
                                ngraph::helpers::ReductionType::LogicalOr,
                                ngraph::helpers::ReductionType::LogicalAnd,
                                ngraph::helpers::ReductionType::L2),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::ValuesIn(layouts4D),
                testing::Values(std::vector<size_t> {1, 512, 7, 7}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
            KmbReduceOpsLayerWithSpecificInputTest::getTestCaseName
    );

    // Test hangs on x86 when executes test-case
    // smoke_Reduce_Axes/KmbReduceOpsLayerTest.CompareWithRefs/IS=(10.20.30.40)_axes=(1)_opType=VECTOR_
    // type=Mean_KeepDims_netPRC=FP32_inPRC=UNSPECIFIED_outPRC=UNSPECIFIED_inL=ANY_trgDev=KMB
    // [Track number: S#43428]
    INSTANTIATE_TEST_SUITE_P(
            smoke_Reduce_Axes_SKIP_MCM,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::ValuesIn(axes),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::ValuesIn(keepDims),
                testing::Values(ngraph::helpers::ReductionType::Mean),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)
            ),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // [Track number: E#22737]
    // Full Reduce currently only works with keepDims=true, since SCALAR output is not supported
    INSTANTIATE_TEST_SUITE_P(
            smoke_Reduce_Axes_Full_SKIP_MCM,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{0, 1, 2, 3}}),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true),
                testing::Values(ngraph::helpers::ReductionType::Mean),
                testing::ValuesIn(netPrecisions),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                testing::Values(InferenceEngine::Layout::ANY),
                testing::Values(std::vector<size_t>{2, 2, 2, 2}),
                testing::Values(LayerTestsUtils::testPlatformTargetDevice)
            ),
            KmbReduceOpsLayerTest::getTestCaseName
    );

    // {2, 4, 6, 8} case is replaced with Reshape to {4, 6, 8}, that MCM fails to compile
    // Tensor:Parameter_375:0 - ArgumentError: attribute identifier allocators - Undefined identifier
    INSTANTIATE_TEST_SUITE_P(
            smoke_ReduceOneAxis_SKIP_MCM,
            KmbReduceOpsLayerTest,
            testing::Combine(
                testing::ValuesIn(decltype(axes) {{0}, {1}}),
                testing::Values(CommonTestUtils::OpType::VECTOR),
                testing::Values(true, false),
                testing::Values(ngraph::helpers::ReductionType::Mean,
                                ngraph::helpers::ReductionType::Max),
                testing::ValuesIn(netPrecisions),
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
                    testing::Values(CommonTestUtils::OpType::VECTOR),
                    testing::Values(true, false),
                    testing::Values(ngraph::helpers::ReductionType::Mean,
                                    ngraph::helpers::ReductionType::Max),
                    testing::ValuesIn(netPrecisions),
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
            smoke_Reduce_Sum_from_networks_SKIP_MCM,
            KmbReduceOpsLayerTest,
            testing::Combine(
                    testing::ValuesIn(decltype(axes) { {2, 3} }),
                    testing::Values(CommonTestUtils::OpType::VECTOR),
                    testing::Values(true, false),
                    testing::Values(ngraph::helpers::ReductionType::Sum),
                    testing::ValuesIn(netPrecisions),
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
            smoke_Reduce_from_networks_SKIP_MCM,
            KmbReduceOpsLayerTest,
            testing::Combine(
                    testing::ValuesIn(decltype(axes) { {2, 3} }),
                    testing::Values(CommonTestUtils::OpType::VECTOR),
                    testing::Values(true, false),
                    testing::Values(ngraph::helpers::ReductionType::Mean),
                    testing::ValuesIn(netPrecisions),
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
