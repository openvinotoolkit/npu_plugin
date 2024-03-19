// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/gather_tree.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class GatherTreeLayerTestCommon : public GatherTreeLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SkipBeforeLoad() override {
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::InputLayerType secondaryInputType;
        std::tie(std::ignore, secondaryInputType, netPrecision, std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore) = GetParam();

        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            throw LayerTestsUtils::VpuSkipTestException("Unsupported secondaryInputType, OV provides scalor end_token "
                                                        "only, but plugin only supports tensors.");
        }

        if (netPrecision == InferenceEngine::Precision::FP32 &&
            secondaryInputType == ngraph::helpers::InputLayerType::CONSTANT) {
            throw LayerTestsUtils::VpuSkipTestException(
                    "FP32 precision with secondaryInputType == CONSTANT generates invalid parent_ids!");
        }
    }
};

class GatherTreeLayerTest_NPU3720 : public GatherTreeLayerTestCommon {};

TEST_P(GatherTreeLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<size_t>> inShapes = {
        {10, 1, 100},
        {5, 1, 10},
        {3, 2, 3},
        {20, 20, 10},
};

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {ngraph::helpers::InputLayerType::CONSTANT,
                                                                          ngraph::helpers::InputLayerType::PARAMETER};

const std::vector<InferenceEngine::Precision> netPrecision = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32, InferenceEngine::Precision::FP16};

const std::vector<InferenceEngine::Precision> precision = {InferenceEngine::Precision::UNSPECIFIED};

const std::vector<InferenceEngine::Layout> layouts = {InferenceEngine::Layout::ANY};

const auto gatherTreeArgsSubsetPrecommit = testing::Combine(
        testing::ValuesIn(inShapes), testing::ValuesIn(secondaryInputTypes), testing::ValuesIn(netPrecision),
        testing::ValuesIn(precision), testing::ValuesIn(precision), testing::ValuesIn(layouts),
        testing::ValuesIn(layouts), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(precommit_gather_tree, GatherTreeLayerTest_NPU3720, gatherTreeArgsSubsetPrecommit,
                         GatherTreeLayerTest::getTestCaseName);

}  // namespace
