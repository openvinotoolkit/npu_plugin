//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace ov::test {
using LargeMinimumTestParams = std::tuple<ov::Shape>;

class LargeMinimumTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<LargeMinimumTestParams> {
    void SetUp() override {
        auto inputShape = std::get<ov::Shape>(GetParam());

        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const auto input1 = ngraph::builder::makeConstant<float>(ov::element::f16, inputShape, {1.0f}, false);
        auto minimum = std::make_shared<ov::op::v1::Minimum>(params[0], input1);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(minimum)};
        function = std::make_shared<ov::Model>(results, params, "LargeMinimumTest");
    }
};

TEST_P(LargeMinimumTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

INSTANTIATE_TEST_CASE_P(smoke_LargeMinimumInputsInDDR, LargeMinimumTest_NPU3720,
                        ::testing::Values(LargeMinimumTestParams{
                                {1, 32, 32, 514}  // in_shape
                        }));

INSTANTIATE_TEST_CASE_P(smoke_LargeMinimumIOInDDR, LargeMinimumTest_NPU3720,
                        ::testing::Values(LargeMinimumTestParams{
                                {64, 32, 32, 16}  // in_shape
                        }));
}  // namespace ov::test
