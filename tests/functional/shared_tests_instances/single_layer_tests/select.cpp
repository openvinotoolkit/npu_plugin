//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "single_layer_tests/select.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {
class SelectLayerTestCommon : public SelectLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        std::vector<std::vector<size_t>> inputShapes(3);
        InferenceEngine::Precision inputPrecision;
        ngraph::op::AutoBroadcastSpec broadcast;
        std::tie(inputShapes, inputPrecision, broadcast, targetDevice) = this->GetParam();

        auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
        ov::ParameterVector inputs;
        for (auto&& shape : inputShapes) {
            inputs.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        ngraph::OutputVector selectInputs;
        auto boolInput = std::make_shared<ov::op::v0::Convert>(inputs[0], ngraph::element::boolean);
        selectInputs.push_back(boolInput);
        for (size_t i = 1; i < inputShapes.size(); i++) {
            selectInputs.push_back(inputs[i]);
        }

        auto select =
                std::dynamic_pointer_cast<ov::op::v1::Select>(ngraph::builder::makeSelect(selectInputs, broadcast));
        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(select)};
        function = std::make_shared<ngraph::Function>(results, inputs, "select");
    }
};

class SelectLayerTest_NPU3700 : public SelectLayerTestCommon {};
class SelectLayerTest_NPU3720 : public SelectLayerTestCommon {};

TEST_P(SelectLayerTest_NPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(SelectLayerTest_NPU3720, SW) {
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

const std::vector<std::vector<std::vector<size_t>>> shapesHighDims = {
        {{2, 3, 4, 5}, {2, 3, 4, 5}, {2, 3, 4, 5}},
        {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}},
};

const std::vector<std::vector<std::vector<size_t>>> inShapes = {{{10, 2, 1, 1}, {10, 2, 1, 1}, {1, 2, 1, 1}}};

const auto selectTestParams0 = ::testing::Combine(::testing::ValuesIn(shapes), ::testing::ValuesIn(inputPrecision),
                                                  ::testing::Values(ov::op::AutoBroadcastType::NONE),
                                                  ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto selectTestParams_highDims =
        ::testing::Combine(::testing::ValuesIn(shapesHighDims), ::testing::ValuesIn(inputPrecision),
                           ::testing::Values(ov::op::AutoBroadcastType::NONE),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto selectTestParams1 = ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision),
                                                  ::testing::Values(ov::op::AutoBroadcastType::NUMPY),
                                                  ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// --------- NPU3700 ---------

INSTANTIATE_TEST_CASE_P(smoke_select_common, SelectLayerTest_NPU3700, selectTestParams0,
                        SelectLayerTest_NPU3700::getTestCaseName);

// Tracking number [E#85137]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_select_highDims, SelectLayerTest_NPU3700, selectTestParams_highDims,
                        SelectLayerTest_NPU3700::getTestCaseName);

// --------- NPU3720 ---------

INSTANTIATE_TEST_CASE_P(smoke_Select, SelectLayerTest_NPU3720, selectTestParams1,
                        SelectLayerTest_NPU3720::getTestCaseName);

}  // namespace
