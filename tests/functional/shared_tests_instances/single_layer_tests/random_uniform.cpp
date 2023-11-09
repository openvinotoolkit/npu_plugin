//
// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/random_uniform.hpp"
#include <ie_precision.hpp>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXRandomLayerTest : public RandomUniformLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void ConfigureNetwork() override {
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    }

    // OpenVino 'SetUp' builds the test-ngraph without any Parameter (since inputs are Constants) => Exception:
    // "/vpux-plugin/src/vpux_imd_backend/src/infer_request.cpp:73 No information about network's output/input"
    // So cloning locally 'SetUp' and providing a 'dummy' Parameter
    template <InferenceEngine::Precision::ePrecision p>
    std::shared_ptr<ov::op::v0::Constant> createRangeConst(
            const typename InferenceEngine::PrecisionTrait<p>::value_type& value) {
        return std::make_shared<ov::op::v0::Constant>(
                FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(p), ov::Shape{1},
                std::vector<typename InferenceEngine::PrecisionTrait<p>::value_type>{value});
    }

    std::shared_ptr<ov::op::v0::Constant> createConstant(InferenceEngine::Precision p, double value) {
        using namespace InferenceEngine;
        switch (p) {
        case Precision::FP32:
            return createRangeConst<InferenceEngine::Precision::FP32>(
                    static_cast<PrecisionTrait<Precision::FP32>::value_type>(value));
        case Precision::FP16:
            return createRangeConst<InferenceEngine::Precision::FP16>(
                    static_cast<PrecisionTrait<Precision::FP16>::value_type>(value));
        default:
            return createRangeConst<InferenceEngine::Precision::I32>(
                    static_cast<PrecisionTrait<Precision::I32>::value_type>(value));
        }
    }

    void SetUp() override {
        RandomUniformTypeSpecificParams randomUniformParams;
        int64_t global_seed;
        int64_t op_seed;
        ov::Shape output_shape;
        std::string targetName;

        std::tie(output_shape, randomUniformParams, global_seed, op_seed, targetDevice) = this->GetParam();
        std::vector<size_t> dummy_shape{1};

        const auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(randomUniformParams.precision);
        auto dummy_params = ngraph::builder::makeParams(precision, {dummy_shape});
        auto out_shape_ =
                std::make_shared<ov::opset8::Constant>(ov::element::i64, ov::Shape{output_shape.size()}, output_shape);

        auto min_value = createConstant(randomUniformParams.precision, randomUniformParams.min_value);
        auto max_value = createConstant(randomUniformParams.precision, randomUniformParams.max_value);
        auto random_uniform = std::make_shared<ngraph::op::v8::RandomUniform>(out_shape_, min_value, max_value,
                                                                              precision, global_seed, op_seed);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(random_uniform)};

        function = std::make_shared<ngraph::Function>(results, dummy_params, "random_uniform");
    }
};

class VPUXRandomLayerTest_VPU3720 : public VPUXRandomLayerTest {};

TEST_P(VPUXRandomLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<RandomUniformTypeSpecificParams> random_uniform_type_specific_params = {
        {InferenceEngine::Precision::FP16, 0.0f, 1.0f},
        {InferenceEngine::Precision::FP16, -10.0, 10.0},
        {InferenceEngine::Precision::I32, -20, 90}};

const std::vector<int64_t> global_seeds = {0, 3456};
const std::vector<int64_t> op_seeds = {11, 876};

const std::vector<ov::Shape> output_shapes = {{1, 200}, {1, 4, 64, 64}};

const auto randParams =
        ::testing::Combine(::testing::ValuesIn(output_shapes), ::testing::ValuesIn(random_uniform_type_specific_params),
                           ::testing::ValuesIn(global_seeds), ::testing::ValuesIn(op_seeds),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_RandomUniform, VPUXRandomLayerTest_VPU3720, randParams,
                         VPUXRandomLayerTest_VPU3720::getTestCaseName);

}  // namespace
