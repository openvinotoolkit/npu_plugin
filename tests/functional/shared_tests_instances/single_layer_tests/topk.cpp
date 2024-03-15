//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/topk.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class TopKLayerTestCommon : virtual public TopKLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class TopKLayerTest_NPU3700 : public TopKLayerTestCommon {};
class TopKLayerTest_NPU3720 : public TopKLayerTestCommon {};

TEST_P(TopKLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(TopKLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

class TopK1LayerTest_NPU3720 : public TopKLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout;
        int64_t keepK, axis;
        ov::op::v3::TopK::Mode mode;
        ov::op::v3::TopK::SortType sort;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
                this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto paramIn =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        auto k = std::make_shared<ov::op::v0::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
        auto topk = std::dynamic_pointer_cast<ov::op::v3::TopK>(
                std::make_shared<ov::op::v3::TopK>(paramIn[0], k, axis, mode, sort));

        ngraph::ResultVector results;
        for (int i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "TopK");
    }
};

TEST_P(TopK1LayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<int64_t> axes = {0, 1, 2};

const std::vector<int64_t> k = {1, 5, 10};

const std::vector<ov::op::v3::TopK::Mode> modes = {ov::op::v3::TopK::Mode::MIN, ov::op::v3::TopK::Mode::MAX};

const std::vector<ov::op::v3::TopK::SortType> sortTypes = {
        // The implements of SortType::NONE are different.
        // Reference uses std::nth_element and returns k out-of-order values.
        // Kernel returns k data sorted in values. nth_element causes computation increase.
        // ov::op::v3::TopK::SortType::NONE,
        ov::op::v3::TopK::SortType::SORT_INDICES,
        ov::op::v3::TopK::SortType::SORT_VALUES,
};

const auto paramsConfig = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{1, 5}), ::testing::ValuesIn(axes), ::testing::ValuesIn(modes),
        ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(std::vector<size_t>({5, 5, 5})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsConfigPrecommit = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{5}), ::testing::ValuesIn(std::vector<int64_t>{2}),
        ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(std::vector<size_t>({5, 5, 5})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(smoke_TopK, TopKLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(k), ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({10, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         TopKLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK, TopKLayerTest_NPU3720, paramsConfig, TopKLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK1, TopK1LayerTest_NPU3720, paramsConfig, TopKLayerTest::getTestCaseName);

// Tiling tests
const std::vector<int64_t> k_Tilling = {1};
const std::vector<int64_t> axes_Tilling = {1};
const std::vector<ov::op::v3::TopK::Mode> modes_Tilling = {ov::op::v3::TopK::Mode::MAX};
const std::vector<ov::op::v3::TopK::SortType> sortTypes_Tilling = {
        ov::op::v3::TopK::SortType::SORT_INDICES,
};
const std::vector<InferenceEngine::Precision> netPrecisions_Tilling = {InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_Tilling, TopKLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling), ::testing::ValuesIn(axes_Tilling),
                                            ::testing::ValuesIn(modes_Tilling), ::testing::ValuesIn(sortTypes_Tilling),
                                            ::testing::ValuesIn(netPrecisions_Tilling),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 5, 512, 512})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         TopKLayerTest::getTestCaseName);

}  // namespace
