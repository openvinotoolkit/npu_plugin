//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include "single_layer_tests/select.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class KmbSelectLayerTest : public SelectLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        std::vector<std::vector<size_t>> inputShapes(3);
        InferenceEngine::Precision inputPrecision;
        ngraph::op::AutoBroadcastSpec broadcast;
        std::tie(inputShapes, inputPrecision, broadcast, targetDevice) = this->GetParam();

        auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
        auto inputs = ngraph::builder::makeParams(inType, inputShapes);
        ngraph::OutputVector selectInputs;
        auto boolInput = std::make_shared<ngraph::opset5::Convert>(inputs[0], ngraph::element::boolean);
        selectInputs.push_back(boolInput);
        for (size_t i = 1; i < inputShapes.size(); i++) {
            selectInputs.push_back(inputs[i]);
        }

        auto select =
                std::dynamic_pointer_cast<ngraph::opset1::Select>(ngraph::builder::makeSelect(selectInputs, broadcast));
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(select)};
        function = std::make_shared<ngraph::Function>(results, inputs, "select");
    }
};

class KmbSelectLayerTest_MLIR : public KmbSelectLayerTest {};
class KmbSelectLayerTest_VPU3720 : public KmbSelectLayerTest {};

TEST_P(KmbSelectLayerTest_MLIR, CompareWithRefs_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbSelectLayerTest_VPU3720, SW_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecision = {
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<std::vector<size_t>>> shapes = {
        {{1}, {1}, {1}},
        {{8}, {8}, {8}},
        {{4, 5}, {4, 5}, {4, 5}},
        {{3, 4, 5}, {3, 4, 5}, {3, 4, 5}},
};

const auto selectTestParams = ::testing::Combine(::testing::ValuesIn(shapes), ::testing::ValuesIn(inputPrecision),
                                                 ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE),
                                                 ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const std::vector<std::vector<std::vector<size_t>>> shapesHighDims = {
        {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
        {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
};

const auto selectTestParams_highDims =
        ::testing::Combine(::testing::ValuesIn(shapesHighDims), ::testing::ValuesIn(inputPrecision),
                           ::testing::Values(ngraph::op::AutoBroadcastSpec::NONE),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_select_common, KmbSelectLayerTest_MLIR, selectTestParams,
                        KmbSelectLayerTest::getTestCaseName);

// Blob generated from VPUIP_2 does not generate the expected result when the 4th dimension's is not 1
// [Track number: E#26954]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_select_highDims, KmbSelectLayerTest_MLIR, selectTestParams_highDims,
                        KmbSelectLayerTest::getTestCaseName);

const std::vector<std::vector<std::vector<size_t>>> inShapesVPU3720 = {{{10, 2, 1, 1}, {10, 2, 1, 1}, {1, 2, 1, 1}}};

const auto selectTestParams_VPU3720 =
        ::testing::Combine(::testing::ValuesIn(inShapesVPU3720), ::testing::ValuesIn(inputPrecision),
                           ::testing::Values(ngraph::op::AutoBroadcastSpec::NUMPY),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_Select_VPU3720, KmbSelectLayerTest_VPU3720, selectTestParams_VPU3720,
                        KmbSelectLayerTest::getTestCaseName);

}  // namespace
