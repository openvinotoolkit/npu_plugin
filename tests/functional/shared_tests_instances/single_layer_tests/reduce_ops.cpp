//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reduce_ops.hpp"

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXReduceLayerTest : public ReduceOpsLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
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

        if ((reductionType == ngraph::helpers::ReductionType::LogicalOr) ||
            (reductionType == ngraph::helpers::ReductionType::LogicalAnd)) {
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

            // Boolean type is unsupported in vpux-plugin, this is why the Convert layers are used to convert the input
            // type into ngraph::element::boolean.
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
};

class VPUXReduceOpsLayerTest_VPU3700 : public VPUXReduceLayerTest {};

TEST_P(VPUXReduceOpsLayerTest_VPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXReduceOpsLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

class VPUXReduceOpsLayerWithSpecificInputTest :
        public ReduceOpsLayerWithSpecificInputTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
    }
    void SkipBeforeValidate() override {
    }
};

TEST_P(VPUXReduceOpsLayerWithSpecificInputTest, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

class VPUXReduceSWLayerTest_VPU3720 : public VPUXReduceLayerTest {};
class VPUXReduceHWLayerTest_VPU3720 : public VPUXReduceLayerTest {};

//
// [Track number: E#66858]
// Using fp16 precision instead of fp32, to reduce more time when the strategy is used
class VPUXReduceLayerHWTilingTest_VPU3720 : public VPUXReduceLayerTest {
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const {
        InferenceEngine::Precision netPrecision = std::get<4>(GetParam());
        auto blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        CommonTestUtils::fill_data_random_float<InferenceEngine::Precision::FP16>(blob, 5, 0, 1000);

        return blob;
    }
};

TEST_P(VPUXReduceSWLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXReduceHWLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    Run();
}

TEST_P(VPUXReduceLayerHWTilingTest_VPU3720, HW) {
    setPlatformVPU3720();
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

const std::vector<std::vector<int>> axes = {{0},    {1},    {2},       {3},       {0, 1},    {0, 2},    {0, 3}, {1, 2},
                                            {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}, {1, -1}};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
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

// [Track number: S#43428]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ReduceOneAxis, VPUXReduceOpsLayerTest_VPU3700,
                         testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::ValuesIn(opTypes),
                                          testing::Values(true, false), testing::ValuesIn(reductionTypes),
                                          testing::ValuesIn(netPrecisions),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::ValuesIn(inputShapesOneAxis),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXReduceOpsLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_smoke_Reduce, VPUXReduceOpsLayerWithSpecificInputTest,
        testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::Values(CommonTestUtils::OpType::VECTOR),
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
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceOpsLayerWithSpecificInputTest::getTestCaseName);

// [Track number: E#22733]
INSTANTIATE_TEST_SUITE_P(
        DISABLED_smoke_Reduce3D, VPUXReduceOpsLayerWithSpecificInputTest,
        testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::Values(CommonTestUtils::OpType::VECTOR),
                         testing::Values(true),
                         testing::Values(ngraph::helpers::ReductionType::Mean, ngraph::helpers::ReductionType::Min,
                                         ngraph::helpers::ReductionType::L1, ngraph::helpers::ReductionType::LogicalOr,
                                         ngraph::helpers::ReductionType::LogicalAnd,
                                         ngraph::helpers::ReductionType::Prod, ngraph::helpers::ReductionType::L2),
                         testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::ValuesIn(layouts3D),
                         testing::Values(std::vector<size_t>{512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceOpsLayerWithSpecificInputTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_smoke_Reduce4D, VPUXReduceOpsLayerWithSpecificInputTest,
        testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::Values(CommonTestUtils::OpType::VECTOR),
                         testing::Values(true),
                         testing::Values(ngraph::helpers::ReductionType::Mean, ngraph::helpers::ReductionType::Min,
                                         ngraph::helpers::ReductionType::L1, ngraph::helpers::ReductionType::LogicalOr,
                                         ngraph::helpers::ReductionType::LogicalAnd,
                                         ngraph::helpers::ReductionType::L2),
                         testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::ValuesIn(layouts4D),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceOpsLayerWithSpecificInputTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_Reduce_from_networks, VPUXReduceOpsLayerTest_VPU3700,
        testing::Combine(testing::ValuesIn(decltype(axes){{2, 3}}), testing::Values(CommonTestUtils::OpType::VECTOR),
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
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceOpsLayerTest_VPU3700::getTestCaseName);

//
// VPU3720 Instantiation
//

const std::vector<InferenceEngine::Precision> netPrecisionsVPUX = {InferenceEngine::Precision::FP16};

const std::vector<ngraph::helpers::ReductionType> reduceOperationsVPUX = {
        ngraph::helpers::ReductionType::Mean,      ngraph::helpers::ReductionType::Max,
        ngraph::helpers::ReductionType::Min,       ngraph::helpers::ReductionType::Sum,
        ngraph::helpers::ReductionType::LogicalOr, ngraph::helpers::ReductionType::LogicalAnd,
        ngraph::helpers::ReductionType::L1,
};

const std::vector<bool> keepDimsVPUX = {
        true,
        false,
};

const std::vector<std::vector<int>> axesVPUX = {{2}, {0, 1}, {0, 1, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_MLIR_VPU3720, VPUXReduceSWLayerTest_VPU3720,
                         testing::Combine(testing::ValuesIn(axesVPUX), testing::Values(CommonTestUtils::OpType::VECTOR),
                                          testing::ValuesIn(keepDimsVPUX), testing::ValuesIn(reduceOperationsVPUX),
                                          testing::ValuesIn(netPrecisionsVPUX),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{1, 512, 7, 7}),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXReduceSWLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Reduce_MLIR_VPU3720, VPUXReduceSWLayerTest_VPU3720,
        testing::Combine(testing::ValuesIn(decltype(axes){{1, -1}}), testing::Values(CommonTestUtils::OpType::VECTOR),
                         testing::ValuesIn(keepDimsVPUX),
                         testing::Values(ngraph::helpers::ReductionType::Mean, ngraph::helpers::ReductionType::Max,
                                         ngraph::helpers::ReductionType::Min, ngraph::helpers::ReductionType::LogicalOr,
                                         ngraph::helpers::ReductionType::LogicalAnd,
                                         ngraph::helpers::ReductionType::L1),
                         testing::ValuesIn(netPrecisionsVPUX), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceSWLayerTest_VPU3720::getTestCaseName);

// Track number [E#69805]
INSTANTIATE_TEST_SUITE_P(
        DISABLED_TMP_smoke_precommit_Reduce_MLIR_VPU3720, VPUXReduceSWLayerTest_VPU3720,
        testing::Combine(testing::ValuesIn(decltype(axes){{1, -1}}), testing::Values(CommonTestUtils::OpType::VECTOR),
                         testing::ValuesIn(keepDimsVPUX), testing::Values(ngraph::helpers::ReductionType::Sum),
                         testing::ValuesIn(netPrecisionsVPUX), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceSWLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Reduce_MLIR_VPU3720_Prod_L2, VPUXReduceSWLayerTest_VPU3720,
        testing::Combine(testing::ValuesIn(decltype(axes){{0}}), testing::Values(CommonTestUtils::OpType::VECTOR),
                         testing::ValuesIn(keepDimsVPUX),
                         testing::Values(ngraph::helpers::ReductionType::Prod, ngraph::helpers::ReductionType::L2),
                         testing::ValuesIn(netPrecisionsVPUX), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY),
                         testing::Values(std::vector<size_t>{1, 512, 7, 7}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceSWLayerTest_VPU3720::getTestCaseName);

const std::vector<std::vector<int>> axesVPUXHW = {{1}, {2}};

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_Reduce_MLIR_VPU3720_HW, VPUXReduceHWLayerTest_VPU3720,
        testing::Combine(testing::ValuesIn(axesVPUXHW), testing::Values(CommonTestUtils::OpType::VECTOR),
                         testing::ValuesIn(keepDimsVPUX),
                         testing::Values(ngraph::helpers::ReductionType::Sum, ngraph::helpers::ReductionType::Mean,
                                         ngraph::helpers::ReductionType::Min, ngraph::helpers::ReductionType::Max),
                         testing::ValuesIn(netPrecisionsVPUX), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY),
                         testing::Values(std::vector<size_t>{1, 9, 32, 32}, std::vector<size_t>{1, 1, 2}),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        VPUXReduceHWLayerTest_VPU3720::getTestCaseName);

//
// VPU3720 functional tiling test is currently deactivated
// The ConvertReduceToPoolingPass must be disabled for the tiling strategy to be applied on the ReduceSum operator
// Another motive would be the execution time with MoviSim, which takes ~30 mins
//

const std::vector<std::vector<int>> axesVPU3720Tiling = {{1}};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Reduce_tiling_MLIR_VPU3720, VPUXReduceLayerHWTilingTest_VPU3720,
                         testing::Combine(testing::ValuesIn(axesVPU3720Tiling),
                                          testing::Values(CommonTestUtils::OpType::VECTOR), testing::Values(true),
                                          testing::Values(ngraph::helpers::ReductionType::Sum),
                                          testing::ValuesIn(netPrecisionsVPUX),
                                          testing::Values(InferenceEngine::Precision::FP16),
                                          testing::Values(InferenceEngine::Precision::FP16),
                                          testing::Values(InferenceEngine::Layout::NCHW),
                                          testing::Values(std::vector<size_t>{1, 9, 80, 1280}),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXReduceLayerHWTilingTest_VPU3720::getTestCaseName);

//                 ReduceMax tests for U8 data type resnet-50-pytorch                      //

const std::vector<InferenceEngine::Precision> netPrecisionsVPUXResnet = {InferenceEngine::Precision::U8};

const std::vector<ngraph::helpers::ReductionType> reduceOperationsVPUXResnet = {
        ngraph::helpers::ReductionType::Max,

};

const std::vector<bool> keepDimsVPUXResnet = {
        true,
};

const std::vector<std::vector<int>> axesVPUXResnet = {{2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_Reduce_MLIR_VPU3720_Resnet, VPUXReduceSWLayerTest_VPU3720,
                         testing::Combine(testing::ValuesIn(axesVPUXResnet),
                                          testing::Values(CommonTestUtils::OpType::VECTOR),
                                          testing::ValuesIn(keepDimsVPUXResnet),
                                          testing::ValuesIn(reduceOperationsVPUXResnet),
                                          testing::ValuesIn(netPrecisionsVPUXResnet),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                          testing::Values(InferenceEngine::Layout::ANY),
                                          testing::Values(std::vector<size_t>{1, 2048, 7, 7}),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXReduceSWLayerTest_VPU3720::getTestCaseName);

}  // namespace
