//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/activation.hpp"

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class ActivationLayerTestCommon : public ActivationLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class ActivationLayerTest_NPU3700 : public ActivationLayerTestCommon {};
class ActivationLayerTest_NPU3720 : public ActivationLayerTestCommon {};
using ActivationLayerTest_NPU3720_ELF = ActivationLayerTest_NPU3720;

class ActivationTilingTest_NPU3720 : public ActivationLayerTestCommon {};

TEST_P(ActivationLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ActivationLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(ActivationTilingTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ActivationLayerTest_NPU3720_ELF, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    useELFCompilerBackend();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid, {{1.0f}}},     {Sign, {{1.0f}}},         {Tanh, {{1.0f}}},
        {Sin, {{1.0f}}},         {Cos, {{1.0f}}},          {Relu, {{1.0f}}},
        {Elu, {{1.0f}}},         {Clamp, {{-1.0f, 1.0f}}}, {HSwish, {{1.0f}}},
        {Mish, {{1.0f}}},        {SoftPlus, {{1.0f}}},     {Floor, {{1.0f}}},
        {Sqrt, {{1.0f}}},        {Sinh, {{1.0f}}},         {Cosh, {{1.0f}}},
        {Asinh, {{1.0f}}},       {Acosh, {{1.0f}}},        {Atanh, {{1.0f}}},
        {Erf, {{1.0f}}},         {Gelu, {{1.0f}}},         {Exp, {{1.0f}}},
        {Log, {{1.0f}}},         {Selu, {{1.0f}}},         {Swish, {{1.0f}}},
        {Negative, {{1.0f}}},    {Abs, {{1.0f}}},          {Atan, {{1.0f}}},
        {Asin, {{1.0f}}},        {Acos, {{1.0f}}},         {HSigmoid, {{1.0f}}},
        {HardSigmoid, {{1.0f}}}, {RoundHalfToEven, {}},    {RoundHalfAwayFromZero, {}},
        {Ceiling, {{1.0f}}},     {Tan, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
        {PReLu, {{0.01f}}},
        {LeakyRelu, {{0.01f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesND = {
        {Sigmoid, {{1.0f}}}, {Tanh, {{1.0f}}},         {Sin, {{1.0f}}},    {Cos, {{1.0f}}}, {Relu, {{1.0f}}},
        {Elu, {{1.0f}}},     {Clamp, {{-1.0f, 1.0f}}}, {HSwish, {{1.0f}}}, {Exp, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesFP16Only = {
        {Ceiling, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesTiling = {
        {Sigmoid, {{1.0f}}},      {Elu, {{1.0f}}},       {Sqrt, {{1.0f}}},       {Exp, {{1.0f}}},
        {Clamp, {{-1.0f, 1.0f}}}, {Tanh, {{1.0f}}},      {LeakyRelu, {{0.01f}}}, {Log, {{1.0f}}},
        {Relu, {{1.0f}}},         {Negative, {{0.01f}}}, {Ceiling, {{1.0f}}}};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes2D = {
        {HSigmoid, {{1.0f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesFP32 = {
        {Relu, {{1.0f}}},
        {Log, {{1.0f}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {{{1, 50, 1, 1}, {{}}}, {{1, 128, 1, 1}, {{}}}};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50, 1, 1}, {{1}, {50}}},
        {{1, 128, 1, 1}, {{1}, {128}}},
        {{1, 32, 96, 96}, {{1}, {32}}},
        {{1, 9, 80, 1280}, {{9}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicNDCase = {
        {{1, 50}, {{}}},
        {{1, 128, 1}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic2DShape = {
        {{120, 50}, {{}}}, {{90, 128}, {{}}}, {{21, 30}, {{}}}};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicTiling = {{{1, 8, 80, 1280}, {{}}}};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes)), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto basicPReluCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationParamTypes)), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(preluBasic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto basicNDCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesND)), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(basicNDCase)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// For operations that only support FP16 input values in 'vpuip_2'
const auto basicFP16OnlyCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesFP16Only)),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto basicCases2D = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes2D)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(basic2DShape)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto basicTilingCases = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesTiling)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(basicTiling)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto basicCasesFP32 = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesFP32)),
        ::testing::Values(InferenceEngine::Precision::FP32), ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// ------ NPU3700 ------

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test, ActivationLayerTest_NPU3700, basicCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test_PRelu, ActivationLayerTest_NPU3700, basicPReluCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test_ND, ActivationLayerTest_NPU3700, basicNDCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test_FP16Only, ActivationLayerTest_NPU3700, basicFP16OnlyCases,
                         ActivationLayerTest::getTestCaseName);

// ------ NPU3720 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test, ActivationLayerTest_NPU3720, basicCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_PRelu, ActivationLayerTest_NPU3720, basicPReluCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_2D, ActivationLayerTest_NPU3720, basicCases2D,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_Activation_Test, ActivationTilingTest_NPU3720, basicTilingCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_FP32_Test, ActivationTilingTest_NPU3720, basicCasesFP32,
                         ActivationLayerTest::getTestCaseName);

// ------ ELF ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_PRelu, ActivationLayerTest_NPU3720_ELF, basicPReluCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Test, ActivationLayerTest_NPU3720_ELF, basicCases,
                         ActivationLayerTest::getTestCaseName);

}  // namespace
