// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gather_tree.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXGatherTreeLayerTest : public GatherTreeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        InferenceEngine::Precision netPrecision;
        ngraph::helpers::InputLayerType secondaryInputType;
        std::tie(std::ignore, secondaryInputType, netPrecision, std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore) = GetParam();

        if (secondaryInputType == ngraph::helpers::InputLayerType::PARAMETER) {
            throw LayerTestsUtils::KmbSkipTestException("Unsupported secondaryInputType, OV provides scalor end_token "
                                                        "only, but plugin only supports tensors.");
        }

        if (netPrecision == InferenceEngine::Precision::FP32 &&
            secondaryInputType == ngraph::helpers::InputLayerType::CONSTANT) {
            throw LayerTestsUtils::KmbSkipTestException(
                    "FP32 precision with secondaryInputType == CONSTANT generates invalid parent_ids!");
        }
    }
};

class VPUXGatherTreeLayerTest_VPU3720 : public VPUXGatherTreeLayerTest {};

TEST_P(VPUXGatherTreeLayerTest_VPU3720, HW) {
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
        testing::ValuesIn(layouts), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(precommit_gather_tree_VPU3720, VPUXGatherTreeLayerTest_VPU3720, gatherTreeArgsSubsetPrecommit,
                         GatherTreeLayerTest::getTestCaseName);

}  // namespace
