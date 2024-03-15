//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>
#include "shared_test_classes/single_op/split.hpp"

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vpu_ov2_layer_test.hpp>
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::utils;

namespace ov::test {

class SplitSoftmaxLayerTest_NPU3700 : public SplitLayerTest, public VpuOv2LayerTest {
    void SetUp() override {
        int64_t axis;
        size_t numSplits;
        std::vector<ov::test::InputShape> inputShape;
        std::vector<size_t> outIndices;
        std::tie(numSplits, axis, inType, inputShape, outIndices, std::ignore) = GetParam();
        if (outIndices.empty()) {
            for (size_t i = 0; i < numSplits; ++i) {
                outIndices.push_back(i);
            }
        }
        init_input_shapes(inputShape);

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        auto split = std::dynamic_pointer_cast<ov::op::v1::Split>(
                ngraph::builder::makeSplit(params[0], inType, numSplits, axis));

        if (axis < 0) {
            axis += inputShape[0].second[0].size();
        }
        ov::ResultVector results;
        results.reserve(outIndices.size());
        for (const auto i : outIndices) {
            const auto softMax = std::make_shared<ov::op::v1::Softmax>(split->output(i), axis);
            results.emplace_back(std::make_shared<ov::op::v0::Result>(softMax));
        }
        function = std::make_shared<ov::Model>(results, params, "split");
    }
};

TEST_P(SplitSoftmaxLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16};
const std::vector<std::vector<ov::Shape>> inputShape = {{{1, 6, 12, 24}}};
const std::vector<size_t> outIndices{};

INSTANTIATE_TEST_SUITE_P(smoke_Split, SplitSoftmaxLayerTest_NPU3700,
                         ::testing::Combine(::testing::Values(2, 3), ::testing::Values(-2, 3),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShape)),
                                            ::testing::Values(outIndices), ::testing::Values(DEVICE_NPU)),
                         SplitLayerTest::getTestCaseName);
}  // namespace ov::test
