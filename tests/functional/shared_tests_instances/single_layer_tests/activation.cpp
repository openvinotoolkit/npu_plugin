//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
namespace {
std::set<ngraph::helpers::ActivationTypes> supportedTypesMLIR{
        ngraph::helpers::Relu,
        ngraph::helpers::Sigmoid,
        ngraph::helpers::Sign,
        ngraph::helpers::Clamp,
        ngraph::helpers::SoftPlus,
        ngraph::helpers::Elu,
        ngraph::helpers::HSwish,
        ngraph::helpers::Floor,
        ngraph::helpers::Mish,
        ngraph::helpers::Erf,
        ngraph::helpers::Tanh,
        ngraph::helpers::Sin,
        ngraph::helpers::Cos,
        ngraph::helpers::PReLu,
        ngraph::helpers::LeakyRelu,
        ngraph::helpers::Swish,
        ngraph::helpers::Negative,
        ngraph::helpers::Exp,
        ngraph::helpers::RoundHalfToEven,
        ngraph::helpers::RoundHalfAwayFromZero,
        ngraph::helpers::Sqrt,
        ngraph::helpers::Sinh,
        ngraph::helpers::Cosh,
        ngraph::helpers::Asinh,
        ngraph::helpers::Acosh,
        ngraph::helpers::Atanh,
        ngraph::helpers::Log,
        ngraph::helpers::Selu,
        ngraph::helpers::Ceiling,
        ngraph::helpers::Gelu,
        ngraph::helpers::Abs,
        ngraph::helpers::Atan,
        ngraph::helpers::Asin,
        ngraph::helpers::Acos,
        ngraph::helpers::HSigmoid,
        ngraph::helpers::HardSigmoid,
        ngraph::helpers::Tan,
};

}  // namespace

class KmbActivationLayerTest : public ActivationLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationParam;
        std::tie(activationParam, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
                 std::ignore) = GetParam();

        const auto activationType = activationParam.first;

        if (supportedTypesMLIR.find(activationType) == supportedTypesMLIR.end()) {
            throw LayerTestsUtils::KmbSkipTestException("Experimental compiler doesn't supports activation type " +
                                                        LayerTestsDefinitions::activationNames[activationType] +
                                                        " yet");
        }
    }
};

class KmbActivationLayerTest_VPU3720 : public KmbActivationLayerTest {};
using ActivationLayerTest_VPU3720_ELF = KmbActivationLayerTest_VPU3720;
class KMBActivationTilingTest_VPU3720 : public KmbActivationLayerTest {};

TEST_P(KmbActivationLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbActivationLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

// [Track number: E#26724]
TEST_P(KmbActivationLayerTest_VPU3720, SW_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KMBActivationTilingTest_VPU3720, HW_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ActivationLayerTest_VPU3720_ELF, SW_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    useELFCompilerBackend();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::FP32};

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
#if 0  // Unsupported layers
    {Sign,     {{1.0f}}},
#endif
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

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {{{1, 50, 1, 1}, {{}}}, {{1, 128, 1, 1}, {{}}}};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50, 1, 1}, {{1}, {50}}},
        {{1, 128, 1, 1}, {{1}, {128}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicNDCase = {
        {{1, 50}, {{}}},
        {{1, 128, 1}, {{}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto basicPReluCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

const auto basicNDCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesND)), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basicNDCase)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

// For operations that only support FP16 input values in 'vpuip_2'
const auto basicFP16OnlyCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesFP16Only)),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Activation_Test, KmbActivationLayerTest, basicCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Activation_Test_PRelu, KmbActivationLayerTest, basicPReluCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Activation_Test_ND, KmbActivationLayerTest, basicNDCases,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Activation_Test_FP16Only, KmbActivationLayerTest, basicFP16OnlyCases,
                         ActivationLayerTest::getTestCaseName);

// ------ VPU3720 ------

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesVPU3720 = {
        {Sigmoid, {{1.0f}}}, {HardSigmoid, {{1.0f}}},  {HSwish, {{1.0f}}},   {RoundHalfToEven, {{1.0f}}},
        {Elu, {{1.0f}}},     {Sqrt, {{1.0f}}},         {Exp, {{1.0f}}},      {RoundHalfAwayFromZero, {{1.0f}}},
        {Mish, {{1.0f}}},    {Clamp, {{-1.0f, 1.0f}}}, {Tanh, {{1.0f}}},     {Selu, {{1.0f}}},
        {Relu, {{1.0f}}},    {Sin, {{1.0f}}},          {Sinh, {{1.0f}}},     {Cosh, {{1.0f}}},
        {Log, {{1.0f}}},     {Erf, {{1.0f}}},          {Tan, {{1.0f}}},      {Floor, {{1.0f}}},
        {Swish, {{1.0f}}},   {Negative, {{1.0f}}},     {Ceiling, {{1.0f}}},  {HSigmoid, {{1.0f}}},
        {Abs, {{1.0f}}},     {Sign, {{1.0f}}},         {SoftPlus, {{1.0f}}}, {Gelu, {{1.0f}}},
        {Cos, {{1.0f}}},     {Asin, {{1.0f}}},         {Acos, {{1.0f}}},     {Asinh, {{1.0f}}},
        {Acosh, {{1.0f}}},   {Atan, {{1.0f}}},         {Atanh, {{1.0f}}}};

const auto basicCasesVPU3720 = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesVPU3720)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_Activation_Test, KmbActivationLayerTest_VPU3720,
                         basicCasesVPU3720, ActivationLayerTest::getTestCaseName);

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasicVPU3720 = {{{1, 50, 1, 1}, {{50}}}};

const auto basicPReluCasesVPU3720 = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasicVPU3720)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_PRelu, KmbActivationLayerTest_VPU3720, basicPReluCasesVPU3720,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_SoftPlus, KmbActivationLayerTest_VPU3720, basicCasesVPU3720,
                         ActivationLayerTest::getTestCaseName);

// ------ Test tiling functionality ------

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesTilingVPU3720 = {
        {Sigmoid, {{1.0f}}},      {Elu, {{1.0f}}},       {Sqrt, {{1.0f}}},       {Exp, {{1.0f}}},
        {Clamp, {{-1.0f, 1.0f}}}, {Tanh, {{1.0f}}},      {LeakyRelu, {{0.01f}}}, {Log, {{1.0f}}},
        {Relu, {{1.0f}}},         {Negative, {{0.01f}}}, {Ceiling, {{1.0f}}}};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basicTiling = {{{1, 8, 80, 1280}, {{}}}};

const auto basicTilingCasesVPU3720 = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesTilingVPU3720)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basicTiling)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_tiling_Activation_Test, KMBActivationTilingTest_VPU3720, basicTilingCasesVPU3720,
                         ActivationLayerTest::getTestCaseName);

// ------ ELF ------

// The ELF test uses custom Activation Type list because the default list includes
// Mish and Swish sw layers that are currently failing
// Issue is tracked in E#64328
const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypesVPU3720_ELF = {
        {Sigmoid, {{1.0f}}},      {HardSigmoid, {{1.0f}}}, {HSwish, {{1.0f}}}, {RoundHalfToEven, {{1.0f}}},
        {Elu, {{1.0f}}},          {Sqrt, {{1.0f}}},        {Exp, {{1.0f}}},    {RoundHalfAwayFromZero, {{1.0f}}},
        {Clamp, {{-1.0f, 1.0f}}}, {Tanh, {{1.0f}}},        {Selu, {{1.0f}}},   {Relu, {{1.0f}}},
        {Sin, {{1.0f}}},          {Sinh, {{1.0f}}},        {Cosh, {{1.0f}}},   {Log, {{1.0f}}},
        {Erf, {{1.0f}}},          {Tan, {{1.0f}}},         {Floor, {{1.0f}}},  {Negative, {{1.0f}}},
        {Ceiling, {{1.0f}}},      {HSigmoid, {{1.0f}}},    {Abs, {{1.0f}}},    {Sign, {{1.0f}}},
        {Cos, {{1.0f}}},          {Asin, {{1.0f}}},        {Acos, {{1.0f}}},   {Asinh, {{1.0f}}},
        {Acosh, {{1.0f}}},        {Atan, {{1.0f}}},        {Atanh, {{1.0f}}}};

const auto basicCasesVPU3720_ELF = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypesVPU3720_ELF)),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Activation_Test_PRelu, ActivationLayerTest_VPU3720_ELF, basicPReluCasesVPU3720,
                         ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Activation_Test, ActivationLayerTest_VPU3720_ELF, basicCasesVPU3720_ELF,
                         ActivationLayerTest::getTestCaseName);

}  // namespace
