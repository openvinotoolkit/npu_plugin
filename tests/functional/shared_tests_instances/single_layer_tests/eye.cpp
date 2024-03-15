//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "shared_test_classes/single_layer/eye.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

// Layer setup with:
// - rows -> Constant
// - cols -> Constant
// - diag_shift -> Parameter
// - batch_shape -> Constant
class EyeLayerTestCommon : public EyeLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    std::vector<ov::Shape> inputShapes;
    std::vector<int> outBatchShape;
    std::vector<int> eyeParams;
    ElementType ngPrc;

    int32_t rowNum, colNum, shift;

    void SetUp() override {
        std::tie(inputShapes, outBatchShape, eyeParams, ngPrc, targetDevice) = GetParam();

        rowNum = eyeParams[0];
        colNum = eyeParams[1];
        shift = eyeParams[2];

        const auto rowsConst = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, inputShapes[0], &rowNum);
        rowsConst->set_friendly_name("rows_const");
        const auto colsConst = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, inputShapes[1], &colNum);
        colsConst->set_friendly_name("cols_const");
        const auto diagShiftPar = std::make_shared<ngraph::opset1::Parameter>(ElementType::i32, inputShapes[2]);

        std::shared_ptr<ngraph::op::v9::Eye> eyeOp;
        if (outBatchShape.empty()) {
            eyeOp = std::make_shared<ngraph::op::v9::Eye>(rowsConst, colsConst, diagShiftPar, ngPrc);
        } else {
            const auto batchShapeConst = std::make_shared<ov::op::v0::Constant>(
                    ngraph::element::i32, ov::Shape{outBatchShape.size()}, outBatchShape.data());
            eyeOp = std::make_shared<ngraph::op::v9::Eye>(rowsConst, colsConst, diagShiftPar, batchShapeConst, ngPrc);
        }
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eyeOp)};

        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{diagShiftPar}, "eye");
    }
    void GenerateInputs() override {
        inputs.clear();
        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        for (size_t i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
            InferenceEngine::InputInfo::CPtr info = infoIt->second;
            auto blob = GenerateInput(*info);

            blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), 1, shift);

            inputs.push_back(blob);
        }
    }
};

// Layer setup with:
// - rows -> Constant
// - cols -> Constant
// - diag_shift -> Constant
// - batch_shape -> Constant
// With OV constant folding (enabled by default), this layer will be calculated by CPU and replaced to Constant operator
class EyeLayerTestWithConstantFoldingCommon :
        public EyeLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    std::vector<ov::Shape> inputShapes;
    std::vector<int> outBatchShape;
    std::vector<int> eyeParams;
    ElementType ngPrc;

    int32_t rowNum, colNum, shift;

    void SetUp() override {
        std::tie(inputShapes, outBatchShape, eyeParams, ngPrc, targetDevice) = GetParam();

        rowNum = eyeParams[0];
        colNum = eyeParams[1];
        shift = eyeParams[2];

        const auto rowsConst = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, inputShapes[0], &rowNum);
        rowsConst->set_friendly_name("rows_const");
        const auto colsConst = std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, inputShapes[1], &colNum);
        colsConst->set_friendly_name("cols_const");
        const auto diagShiftConst =
                std::make_shared<ov::op::v0::Constant>(ngraph::element::i32, inputShapes[2], &shift);
        diagShiftConst->set_friendly_name("diag_shift_const");

        std::shared_ptr<ngraph::op::v9::Eye> eyeOp;
        if (outBatchShape.empty()) {
            eyeOp = std::make_shared<ngraph::op::v9::Eye>(rowsConst, colsConst, diagShiftConst, ngPrc);
        } else {
            const auto batchShapeConst = std::make_shared<ov::op::v0::Constant>(
                    ngraph::element::i32, ov::Shape{outBatchShape.size()}, outBatchShape.data());
            eyeOp = std::make_shared<ngraph::op::v9::Eye>(rowsConst, colsConst, diagShiftConst, batchShapeConst, ngPrc);
        }

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eyeOp)};
        // TODO: E#92001 Resolve the dependency of dummy parameter for layer with all constant inputs
        // Add a dummy parameter input as a workaround to 'No information about network's output/input.' failure when
        // running SLT with IMD backend
        const auto dummyPar = std::make_shared<ngraph::opset1::Parameter>(ElementType::i32, ov::Shape{1});
        dummyPar->set_friendly_name("dummy");

        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{dummyPar}, "eye");
    }
};

class EyeLayerTest_NPU3720 : public EyeLayerTestCommon {};

class EyeLayerTestWithConstantFolding_NPU3720 : public EyeLayerTestWithConstantFoldingCommon {};

TEST_P(EyeLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(EyeLayerTestWithConstantFolding_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// Shape for 'rows', 'cols', and 'diag_shift'
const std::vector<ov::Shape> eyeShape = {{1}, {1}, {1}};

const std::vector<std::vector<int>> batchShapes = {
        {},        // No 'batch_shape' -> output shape = 2D
        {2},       // 1D 'batch_shape' -> output shape = 3D
        {3, 2},    // 2D 'batch_shape' -> output shape = 4D
        {4, 3, 2}  // 3D 'batch_shape' -> output shape = 5D
};

const std::vector<std::vector<int>> eyePars = {
        // rows, cols, diag_shift
        {8, 2, 1},
        {9, 4, 6},
        {5, 7, -3}};

const std::vector<ElementType> netPrecisions = {ElementType::f32, ElementType::f16, ElementType::i32, ElementType::i8,
                                                ElementType::u8};

const auto noBatchShapeParams = testing::Combine(testing::Values(eyeShape), testing::Values(batchShapes[0]),
                                                 testing::ValuesIn(eyePars), testing::ValuesIn(netPrecisions),
                                                 testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto withBatchShapeParams =
        testing::Combine(testing::Values(eyeShape),
                         testing::ValuesIn(std::vector<std::vector<int>>(batchShapes.begin() + 1, batchShapes.end())),
                         testing::Values(eyePars[0]), testing::Values(netPrecisions[0]),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto realNetParams = testing::Combine(
        testing::Values(eyeShape), testing::Values(batchShapes[0]), testing::Values(std::vector<int>{128, 128, 0}),
        testing::Values(netPrecisions[0]), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

/* ============= NPU 3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Eye, EyeLayerTest_NPU3720, noBatchShapeParams, EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_with_batch_shape, EyeLayerTest_NPU3720, withBatchShapeParams,
                         EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_real_net, EyeLayerTest_NPU3720, realNetParams, EyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Eye_const_fold_real_net, EyeLayerTestWithConstantFolding_NPU3720, realNetParams,
                         EyeLayerTest::getTestCaseName);

}  // namespace
