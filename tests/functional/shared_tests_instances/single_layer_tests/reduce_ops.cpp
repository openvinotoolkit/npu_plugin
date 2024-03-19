//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/reduce_ops.hpp"

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"
#include "vpux_private_properties.hpp"

namespace LayerTestsDefinitions {
class ReduceLayerTestCommon : public ReduceOpsLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        ov::test::utils::OpType opType;
        ngraph::helpers::ReductionType reductionType;
        ngraph::NodeVector convertedInputs;
        ngraph::OutputVector paramOuts;
        std::shared_ptr<ngraph::Node> reduceNode;
        std::vector<size_t> inputShape, shapeAxes;
        std::vector<int> axes;
        bool keepDims;

        std::tie(axes, opType, keepDims, reductionType, netPrecision, inPrc, outPrc, inLayout, inputShape,
                 targetDevice) = GetParam();

        if ((reductionType == ngraph::helpers::ReductionType::LogicalOr) ||
            (reductionType == ngraph::helpers::ReductionType::LogicalAnd)) {
            auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
            ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

            switch (opType) {
            case ov::test::utils::OpType::SCALAR: {
                if (axes.size() > 1)
                    FAIL() << "In reduce op if op type is scalar, 'axis' input's must contain 1 element";
                break;
            }
            case ov::test::utils::OpType::VECTOR: {
                shapeAxes.push_back(axes.size());
                break;
            }
            default:
                FAIL() << "Reduce op doesn't support operation type: " << opType;
            }
            auto reductionAxesNode = std::dynamic_pointer_cast<ngraph::Node>(std::make_shared<ov::op::v0::Constant>(
                    ngraph::element::Type_t::i64, ngraph::Shape(shapeAxes), axes));

            // Boolean type is unsupported in npu-plugin, this is why the Convert layers are used to convert the input
            // type into ngraph::element::boolean.
            for (const auto& input : inputs) {
                convertedInputs.push_back(std::make_shared<ov::op::v0::Convert>(input, ngraph::element::boolean));
            }
            paramOuts = ngraph::helpers::convert2OutputVector(convertedInputs);
            reduceNode = ngraph::builder::makeReduce(paramOuts[0], reductionAxesNode, keepDims, reductionType);
            const ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(reduceNode)};
            function = std::make_shared<ngraph::Function>(results, inputs, "Reduce");
        } else {
            ReduceOpsLayerTest::SetUp();
        }
    }
};

// FP16 for 3700 platform
class ReduceOpsLayerTest_NPU3700 : public ReduceLayerTestCommon {};

// FP16/FP32 for 3720 platforms
class ReduceLayerTest_HW_FP16 : public ReduceLayerTestCommon {};
class ReduceLayerTest_SW_FP16 : public ReduceLayerTestCommon {};
class ReduceLayerTest_SW_FP32 : public ReduceLayerTestCommon {
    void ConfigureNetwork() override {
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP32);
        configuration[ov::intel_vpux::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

// NPU3700 SW/HW
TEST_P(ReduceOpsLayerTest_NPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(ReduceOpsLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

/// FP16 SW/HW
// NPU3720
TEST_P(ReduceLayerTest_HW_FP16, NPU3720) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ReduceLayerTest_SW_FP16, NPU3720) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

/// FP32 SW
// NPU3720
TEST_P(ReduceLayerTest_SW_FP32, NPU3720) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

class ReduceOpsLayerWithSpecificInputTestCommon :
        public ReduceOpsLayerWithSpecificInputTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

TEST_P(ReduceOpsLayerWithSpecificInputTestCommon, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

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

const std::vector<std::vector<int>> axes = {{1}, {2}, {1, 3}, {2, 3}, {1, -1}};

std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::SCALAR,
        ov::test::utils::OpType::VECTOR,
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
        ngraph::helpers::ReductionType::Mean,       ngraph::helpers::ReductionType::Min,
        ngraph::helpers::ReductionType::Max,        ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::Prod,       ngraph::helpers::ReductionType::LogicalOr,
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

//
// NPU3700 Instantiation
//

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ReduceOneAxis, ReduceOpsLayerTest_NPU3700,
                         testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::ValuesIn(opTypes),
                                          testing::Values(true, false), testing::ValuesIn(reductionTypes),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::ValuesIn(inputShapesOneAxis),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ReduceOpsLayerTest_NPU3700::getTestCaseName);

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Reduce, ReduceOpsLayerWithSpecificInputTestCommon,
        testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::Values(ov::test::utils::OpType::VECTOR),
                         testing::Values(true, false),
                         testing::Values(ngraph::helpers::ReductionType::Max, ngraph::helpers::ReductionType::Sum,
                                         ngraph::helpers::ReductionType::Min, ngraph::helpers::ReductionType::L1,
                                         ngraph::helpers::ReductionType::LogicalOr,
                                         ngraph::helpers::ReductionType::LogicalAnd,
                                         ngraph::helpers::ReductionType::Prod, ngraph::helpers::ReductionType::L2),
                         testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
        ReduceOpsLayerWithSpecificInputTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Reduce3D, ReduceOpsLayerWithSpecificInputTestCommon,
        testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::Values(ov::test::utils::OpType::VECTOR),
                         testing::Values(true),
                         testing::Values(ngraph::helpers::ReductionType::Mean, ngraph::helpers::ReductionType::Min,
                                         ngraph::helpers::ReductionType::L1, ngraph::helpers::ReductionType::LogicalOr,
                                         ngraph::helpers::ReductionType::LogicalAnd,
                                         ngraph::helpers::ReductionType::Prod, ngraph::helpers::ReductionType::L2),
                         testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::ValuesIn(layouts3D),
                         testing::Values(std::vector<size_t>{512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
        ReduceOpsLayerWithSpecificInputTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Reduce4D, ReduceOpsLayerWithSpecificInputTestCommon,
        testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::Values(ov::test::utils::OpType::VECTOR),
                         testing::Values(true),
                         testing::Values(ngraph::helpers::ReductionType::Mean, ngraph::helpers::ReductionType::Min,
                                         ngraph::helpers::ReductionType::L1, ngraph::helpers::ReductionType::LogicalOr,
                                         ngraph::helpers::ReductionType::LogicalAnd,
                                         ngraph::helpers::ReductionType::L2),
                         testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::ValuesIn(layouts4D),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
        ReduceOpsLayerWithSpecificInputTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Reduce_from_networks, ReduceOpsLayerTest_NPU3700,
        testing::Combine(testing::ValuesIn(decltype(axes){{2, 3}}), testing::Values(ov::test::utils::OpType::VECTOR),
                         testing::Values(true, false),
                         testing::Values(ngraph::helpers::ReductionType::Mean, ngraph::helpers::ReductionType::Max),
                         testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7},   // resnet_18
                                         std::vector<size_t>{1, 2048, 7, 7},  // resnet_50
                                         std::vector<size_t>{1, 1280, 7, 7},  // mobilenet_v2
                                         std::vector<size_t>{1, 1664, 7, 7}   // densenet
                                         ),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
        ReduceOpsLayerTest_NPU3700::getTestCaseName);

//
// NPU3720 Instantiation
//

const std::vector<ngraph::helpers::ReductionType> reduceOperations = {
        ngraph::helpers::ReductionType::Mean,      ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Min,       ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::LogicalOr, ngraph::helpers::ReductionType::LogicalAnd,
        ngraph::helpers::ReductionType::L1,        ngraph::helpers::ReductionType::L2,
        ngraph::helpers::ReductionType::Prod};

//
// FP16 SW
const auto paramsSWFP16 = testing::Combine(
        testing::ValuesIn(axes), testing::Values(ov::test::utils::OpType::VECTOR), testing::ValuesIn(keepDims),
        testing::ValuesIn(reduceOperations), testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 512, 7, 7}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsTiling = testing::Combine(
        testing::ValuesIn(decltype(axes){{2}, {1, -1}}), testing::Values(ov::test::utils::OpType::VECTOR),
        testing::ValuesIn(keepDims), testing::Values(ngraph::helpers::ReductionType::Sum),
        testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(std::vector<size_t>{1, 20, 175, 512}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// ReduceMax config for U8 data type resnet-50-pytorch
const auto paramsResnet = testing::Combine(
        testing::ValuesIn(decltype(axes){{2, 3}}), testing::Values(ov::test::utils::OpType::VECTOR),
        testing::Values(true), testing::Values(ngraph::helpers::ReductionType::Max),
        testing::Values(InferenceEngine::Precision::U8), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 2048, 7, 7}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

auto paramsReduceAllAxisKpDimsTrue = testing::Combine(
        testing::ValuesIn(decltype(axes){{0, 1, 2, 3}}), testing::Values(ov::test::utils::OpType::VECTOR),
        testing::Values(true), testing::ValuesIn(reduceOperations), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 4, 2, 38}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

auto paramsReduceAllAxisKpDimsFalse = testing::Combine(
        testing::ValuesIn(decltype(axes){{0, 1, 2, 3}}), testing::Values(ov::test::utils::OpType::VECTOR),
        testing::Values(false),
        testing::Values(ngraph::helpers::ReductionType::Sum, ngraph::helpers::ReductionType::Min),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 8, 4, 76}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

//
// FP16 HW
const auto paramsHWFP16 = testing::Combine(
        testing::ValuesIn(decltype(axes){{1}, {2}}), testing::Values(ov::test::utils::OpType::VECTOR),
        testing::ValuesIn(keepDims),
        testing::Values(ngraph::helpers::ReductionType::Sum, ngraph::helpers::ReductionType::Mean,
                        ngraph::helpers::ReductionType::Min, ngraph::helpers::ReductionType::Max),
        testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 9, 32, 32}, std::vector<size_t>{1, 1, 2},
                        std::vector<size_t>{1, 4, 32, 32}, std::vector<size_t>{1, 16, 32, 32}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

//
// FP32 SW
const auto paramsSWFP32 = testing::Combine(
        testing::ValuesIn(decltype(axes){{2, 3}}), testing::Values(ov::test::utils::OpType::VECTOR),
        testing::ValuesIn(keepDims),
        testing::Values(ngraph::helpers::ReductionType::Mean, ngraph::helpers::ReductionType::Sum),
        testing::Values(InferenceEngine::Precision::FP32), testing::Values(InferenceEngine::Precision::FP32),
        testing::Values(InferenceEngine::Precision::FP32), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 1024, 7, 7}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

//
// NPU3720 Instantiation
// FP16 HW
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Reduce, ReduceLayerTest_HW_FP16, paramsHWFP16,
                         ReduceLayerTest_HW_FP16::getTestCaseName);

//
// NPU3720 Instantiation
// FP16 SW
INSTANTIATE_TEST_SUITE_P(smoke_Reduce, ReduceLayerTest_SW_FP16, paramsSWFP16, ReduceLayerTest_SW_FP16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_tiling, ReduceLayerTest_SW_FP16, paramsTiling,
                         ReduceLayerTest_SW_FP16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_Resnet, ReduceLayerTest_SW_FP16, paramsResnet,
                         ReduceLayerTest_SW_FP16::getTestCaseName);

// All axes reduced tests
INSTANTIATE_TEST_SUITE_P(smoke_ReduceAllAxis_true, ReduceLayerTest_SW_FP16, paramsReduceAllAxisKpDimsTrue,
                         ReduceLayerTest_SW_FP16::getTestCaseName);
// [Track number E#90461]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ReduceAllAxis_false, ReduceLayerTest_SW_FP16,
                         paramsReduceAllAxisKpDimsFalse, ReduceLayerTest_SW_FP16::getTestCaseName);

// FP32 SW
INSTANTIATE_TEST_SUITE_P(smoke_Reduce_FP32, ReduceLayerTest_SW_FP32, paramsSWFP32,
                         ReduceLayerTest_SW_FP32::getTestCaseName);

}  // namespace
