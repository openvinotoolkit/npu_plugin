//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/fake_quantize.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbFakeQuantizeLayerTest : public FakeQuantizeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            // [Track number: S#42747]
            throw LayerTestsUtils::KmbSkipTestException("Issues with blobs generated with MCM compiler");
        }
    }
};

class KmbFakeQuantizeLayerTest_HW : public KmbFakeQuantizeLayerTest {};

TEST_P(KmbFakeQuantizeLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbFakeQuantizeLayerTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbFakeQuantizeLayerTest_HW, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

const std::vector<std::vector<size_t>> inputShapes = {{1, 3, 10, 10}};
const std::vector<std::vector<size_t>> constShapes = {{1}, {1, 3, 1, 1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::vector<std::vector<size_t>> inputShapesND = {{1, 512}};
const std::vector<std::vector<size_t>> constShapesND = {{1}};

const std::pair<std::string, std::map<std::string, std::string>> config = {};
const std::vector<float> fqArgs = {0, 255, 0, 255};
const std::vector<float> inputParams = {0, 255, 1};

const auto fqParams = ::testing::Combine(
    ::testing::ValuesIn(levels),
    ::testing::ValuesIn(constShapes),
    ::testing::Values(fqArgs),
    ::testing::Values(inputParams),
    ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY)
);

const auto fqParamsND = ::testing::Combine(
    ::testing::ValuesIn(levels),
    ::testing::ValuesIn(constShapesND),
    ::testing::Values(fqArgs),
    ::testing::Values(inputParams),
    ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, KmbFakeQuantizeLayerTest,
                        ::testing::Combine(
                            fqParams,
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(inputShapes),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::Values(config)),
                        KmbFakeQuantizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_ND, KmbFakeQuantizeLayerTest,
                    ::testing::Combine(
                            fqParamsND,
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::ValuesIn(inputShapesND),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::Values(config)),
                    KmbFakeQuantizeLayerTest::getTestCaseName);

// TODO: support levels=16
// "Can't convert 12 Bit to Byte" while working u4 precision (!quant.uniform<u4:f16, 0.5:128>)
const std::vector<size_t> hw_levels = {255, 256};
const auto hw_fqParams = ::testing::Combine(
        ::testing::ValuesIn(hw_levels),
        ::testing::ValuesIn(constShapes),
        ::testing::Values(fqArgs),
        ::testing::Values(inputParams),
        ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY)
);

const auto hw_fqParamsND = ::testing::Combine(
        ::testing::ValuesIn(hw_levels),
        ::testing::ValuesIn(constShapesND),
        ::testing::Values(fqArgs),
        ::testing::Values(inputParams),
        ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, KmbFakeQuantizeLayerTest_HW,
                        ::testing::Combine(
                                hw_fqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                ::testing::Values(config)),
                        KmbFakeQuantizeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_ND, KmbFakeQuantizeLayerTest_HW,
                        ::testing::Combine(
                                hw_fqParamsND,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapesND),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                ::testing::Values(config)),
                        KmbFakeQuantizeLayerTest::getTestCaseName);

}  // namespace
