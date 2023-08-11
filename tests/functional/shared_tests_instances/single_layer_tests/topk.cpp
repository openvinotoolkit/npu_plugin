//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/topk.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXTopKLayerTest : virtual public TopKLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXTopKLayerTest_VPU3700 : public VPUXTopKLayerTest {
    void SkipBeforeLoad() override {
    }
};

class VPUXTopKLayerTest_VPU3720 : public VPUXTopKLayerTest {};

TEST_P(VPUXTopKLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXTopKLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

class VPUXTopK1LayerTest_VPU3720 : public TopKLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        InferenceEngine::SizeVector inputShape;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout;
        int64_t keepK, axis;
        ngraph::opset4::TopK::Mode mode;
        ngraph::opset4::TopK::SortType sort;
        std::tie(keepK, axis, mode, sort, netPrecision, inPrc, outPrc, inLayout, inputShape, targetDevice) =
                this->GetParam();

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramIn =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        auto k = std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{}, &keepK);
        auto topk = std::dynamic_pointer_cast<ngraph::opset1::TopK>(
                std::make_shared<ngraph::opset1::TopK>(paramIn[0], k, axis, mode, sort));

        ngraph::ResultVector results;
        for (int i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ngraph::opset1::Result>(topk->output(i)));
        }
        function = std::make_shared<ngraph::Function>(results, params, "TopK");
    }
};

TEST_P(VPUXTopK1LayerTest_VPU3720, HW) {
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

// reduce the test cases due to mosivim slowness
const std::vector<int64_t> k_VPU3720 = {1, 5};

const std::vector<ngraph::opset4::TopK::Mode> modes = {ngraph::opset4::TopK::Mode::MIN,
                                                       ngraph::opset4::TopK::Mode::MAX};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
        // The implements of SortType::NONE are different.
        // Reference uses std::nth_element and returns k out-of-order values.
        // Kernel returns k data sorted in values. nth_element causes computation increase.
        // ngraph::opset4::TopK::SortType::NONE,
        ngraph::opset4::TopK::SortType::SORT_INDICES,
        ngraph::opset4::TopK::SortType::SORT_VALUES,
};

// [Track number: S#41824]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_TopK, VPUXTopKLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(k), ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({10, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXTopKLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_TopK, VPUXTopKLayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(k_VPU3720), ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({5, 5, 5})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         TopKLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK1, VPUXTopK1LayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(k_VPU3720),
                                            ::testing::ValuesIn(std::vector<int64_t>{0}), ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({5, 5, 5})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         TopKLayerTest::getTestCaseName);

const std::vector<int64_t> k_Tilling = {1};
const std::vector<int64_t> axes_Tilling = {1};
const std::vector<ngraph::opset4::TopK::Mode> modes_Tilling = {ngraph::opset4::TopK::Mode::MAX};
const std::vector<ngraph::opset4::TopK::SortType> sortTypes_Tilling = {
        ngraph::opset4::TopK::SortType::SORT_INDICES,
};
const std::vector<InferenceEngine::Precision> netPrecisions_Tilling = {InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_Tilling, VPUXTopKLayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling), ::testing::ValuesIn(axes_Tilling),
                                            ::testing::ValuesIn(modes_Tilling), ::testing::ValuesIn(sortTypes_Tilling),
                                            ::testing::ValuesIn(netPrecisions_Tilling),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 5, 512, 512})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         TopKLayerTest::getTestCaseName);


}  // namespace
