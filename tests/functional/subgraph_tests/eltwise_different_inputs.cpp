// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace ov::test {

class EltwiseAddQuantizedSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ov::element::Type> {
    void SetUp() override {
        const ov::Shape inputShape{1, 16, 56, 56};
        const ov::Shape weightsShape{1, 16, 56, 56};
        inType = outType = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inputShape, weightsShape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        params[0]->set_friendly_name("input1");
        params[1]->set_friendly_name("input2");

        const size_t dataLevels = 256;
        const auto dataFq = ngraph::builder::makeFakeQuantize(params[0], ov::element::f16, dataLevels, {}, {0.0},
                                                              {12.583984375}, {0.0}, {12.583984375});

        const size_t weightsLevels = 256;
        const auto weightsFq = ngraph::builder::makeFakeQuantize(params[1], ov::element::f16, weightsLevels, {}, {0.0},
                                                                 {2.583984375}, {0.0}, {2.583984375});

        const auto addOp = std::make_shared<ov::op::v1::Add>(dataFq, weightsFq);

        const size_t outLevels = 256;
        const auto outputFq = ngraph::builder::makeFakeQuantize(addOp, ov::element::f16, outLevels, {}, {0.0},
                                                                {13.583984375}, {0.0}, {13.583984375});

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "EltwiseAddQuantized");
        rel_threshold = 0.1f;
    }
};

TEST_P(EltwiseAddQuantizedSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

INSTANTIATE_TEST_CASE_P(smoke_EltwiseAddQuantized, EltwiseAddQuantizedSubGraphTest_NPU3720,
                        ::testing::Values(ov::element::f16));

}  // namespace ov::test
