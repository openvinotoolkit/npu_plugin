//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "single_layer_tests/rdft.hpp"
#include <algorithm>
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class RdftLayerTestCommon : public RDFTLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision inputPrecision;
        std::vector<int64_t> axes;
        std::vector<int64_t> signalSize;
        ngraph::helpers::DFTOpType opType;
        std::tie(inputShapes, inputPrecision, axes, signalSize, opType, targetDevice) = this->GetParam();
        // fp16 preciosion is 0.1 not 0.01 as fp32.
        // And increase for every axes where need intermediate value to keep in fp16
        // Extra increase of precision decrease in fp16 is added by convert to fp16 the precalculated twiddle factors.
        // In this case depend by shape[axes] size. Consider 0.15 cover 64 line width.
        if (inputPrecision == InferenceEngine::Precision::FP16) {
            threshold = 0.15f * axes.size();
        }
        RDFTLayerTest::SetUp();
    }
};

TEST_P(RdftLayerTestCommon, NPU3720) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

namespace {

using namespace LayerTestsDefinitions;

const std::vector<ngraph::helpers::DFTOpType> opTypes = {
        ngraph::helpers::DFTOpType::FORWARD,
        ngraph::helpers::DFTOpType::INVERSE,
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        // disable FP32  tests as default compiler pipelines pass createConvertPrecisionToFP16Pass will convert anyway
        // to fp16 the operation, so test precision will be precision for fp16
        // InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const auto combine = [](const std::vector<InferenceEngine::SizeVector>& inputShapes,
                        const std::vector<std::vector<int64_t>>& axes,
                        const std::vector<std::vector<int64_t>>& signalSizes) {
    return testing::Combine(testing::ValuesIn(inputShapes), testing::ValuesIn(inputPrecisions), testing::ValuesIn(axes),
                            testing::ValuesIn(signalSizes), testing::ValuesIn(opTypes),
                            testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
};

// RDFT can support 1d
INSTANTIATE_TEST_SUITE_P(smoke_RDFT_1d, RdftLayerTestCommon,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10}),
                                          testing::ValuesIn(inputPrecisions), testing::Values(std::vector<int64_t>{0}),
                                          testing::Values(std::vector<int64_t>{}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_2d, RdftLayerTestCommon,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 2}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {3}, {12}}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_2dx, RdftLayerTestCommon,
                         combine({{10, 2}},         // input shapes
                                 {{0}},             // axes
                                 {{}, {3}, {11}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_RDFT_3d, RdftLayerTestCommon,
                         combine({{10, 4, 2}},    // input shapes
                                 {{0, 1}},        // axes
                                 {{}, {3, 10}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d, RdftLayerTestCommon,
                         combine({{10, 4, 8, 2}},    // input shapes
                                 {{0, 1, 2}},        // axes
                                 {{}, {3, 10, 8}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_RDFT_4d_negative_reversed_axes, RdftLayerTestCommon,
                         combine({{10, 4, 8, 2}},    // input shapes
                                 {{-1, -2, -3}},     // axes
                                 {{}, {8, 10, 3}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_4d_single_axis, RdftLayerTestCommon,
                         combine({{10, 4, 8, 2}},        // input shapes
                                 {{0}, {1}, {2}},        // axes
                                 {{}, {1}, {5}, {20}}),  // signal sizes
                         RDFTLayerTest::getTestCaseName);

// IRDFT can support 5d
INSTANTIATE_TEST_SUITE_P(smoke_precommit_RDFT_5d, RdftLayerTestCommon,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{10, 4, 8, 2, 2}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{{0, 1, 2, 3}}}),
                                          testing::ValuesIn(std::vector<std::vector<int64_t>>{{}, {3, 10, 8, 6}}),
                                          testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_tile_FORWARD, RdftLayerTestCommon,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{1, 80, 64, 64}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::Values(std::vector<int64_t>{2, 3}),
                                          testing::Values(std::vector<int64_t>{}),
                                          testing::Values(ngraph::helpers::DFTOpType::FORWARD),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         RDFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RDFT_tile_INVERSE, RdftLayerTestCommon,
                         testing::Combine(testing::Values(InferenceEngine::SizeVector{1, 120, 64, 33, 2}),
                                          testing::ValuesIn(inputPrecisions),
                                          testing::Values(std::vector<int64_t>{2, 3}),
                                          testing::Values(std::vector<int64_t>{}),
                                          testing::Values(ngraph::helpers::DFTOpType::INVERSE),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         RDFTLayerTest::getTestCaseName);

}  // namespace
