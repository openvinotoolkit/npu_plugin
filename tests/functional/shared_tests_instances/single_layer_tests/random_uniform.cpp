//
// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/random_uniform.hpp"
#include <ie_precision.hpp>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"
#include "vpux_private_properties.hpp"

namespace LayerTestsDefinitions {

class RandomLayerTestCommon : public RandomUniformLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void ConfigureNetwork() override {
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    }

    // TODO: E#92001 Resolve the dependency of dummy parameter for layer with all constant inputs
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
        ov::Shape dummy_shape{1};

        const auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(randomUniformParams.precision);
        auto dummy_params = std::make_shared<ov::op::v0::Parameter>(precision, dummy_shape);
        auto out_shape_ =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{output_shape.size()}, output_shape);

        auto min_value = createConstant(randomUniformParams.precision, randomUniformParams.min_value);
        auto max_value = createConstant(randomUniformParams.precision, randomUniformParams.max_value);
        auto random_uniform = std::make_shared<ngraph::op::v8::RandomUniform>(out_shape_, min_value, max_value,
                                                                              precision, global_seed, op_seed);
        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(random_uniform)};

        function = std::make_shared<ngraph::Function>(results, ov::ParameterVector{dummy_params}, "random_uniform");
    }
};

class RandomLayerTest_NPU3720 : public RandomLayerTestCommon {};

TEST_P(RandomLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<RandomUniformTypeSpecificParams> randomUniformSpecificParams = {
        {InferenceEngine::Precision::FP16, 0.0f, 1.0f},
        {InferenceEngine::Precision::FP16, -10.0, 10.0},
        {InferenceEngine::Precision::I32, -20, 90}};

const std::vector<RandomUniformTypeSpecificParams> randomUniformSpecificParamsF32 = {
        {InferenceEngine::Precision::FP32, 0.0f, 1.0f}, {InferenceEngine::Precision::FP32, -10.0, 10.0}};

const std::vector<int64_t> globalSeeds = {0, 3456};
const std::vector<int64_t> opSeeds = {11, 876};

const std::vector<ov::Shape> outputShapes = {{1, 200}, {1, 4, 64, 64}};

const auto randParams =
        ::testing::Combine(::testing::ValuesIn(outputShapes), ::testing::ValuesIn(randomUniformSpecificParams),
                           ::testing::ValuesIn(globalSeeds), ::testing::ValuesIn(opSeeds),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto randParamsF32 =
        ::testing::Combine(::testing::Values(outputShapes[1]), ::testing::ValuesIn(randomUniformSpecificParamsF32),
                           ::testing::Values(globalSeeds[0]), ::testing::Values(opSeeds[1]),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_RandomUniform, RandomLayerTest_NPU3720, randParams,
                         RandomLayerTest_NPU3720::getTestCaseName);

}  // namespace
