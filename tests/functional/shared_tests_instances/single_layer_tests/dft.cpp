//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/dft.hpp"
#include <algorithm>
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class DftLayerTestCommon : public DFTLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        InferenceEngine::SizeVector inputShapes;
        InferenceEngine::Precision inputPrecision;
        std::vector<int64_t> axes;
        std::vector<int64_t> signalSize;
        ngraph::helpers::DFTOpType opType;
        std::tie(inputShapes, inputPrecision, axes, signalSize, opType, targetDevice) = this->GetParam();
        // fp16 precision is 0.1 not 0.01 as fp32.
        // And increase for every axes where need intermediate value to keep in fp16
        // Extra increase of precision decrease in fp16 is added by convert to fp16 the precalculated twiddle factors.
        // In this case depend by shape[axes] size. Consider 0.15 cover 64 line width.
        if (inputPrecision == InferenceEngine::Precision::FP16) {
            threshold = 0.15f * axes.size();
        }
        DFTLayerTest::SetUp();
    }
};

TEST_P(DftLayerTestCommon, NPU3720) {
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

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_2d, DftLayerTestCommon,
                         combine({{10, 2}},   // input shapes
                                 {{0}},       // axes
                                 {{}, {3}}),  // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_3d, DftLayerTestCommon,
                         combine({{10, 4, 2}},    // input shapes
                                 {{0, 1}},        // axes
                                 {{}, {3, 10}}),  // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_4d, DftLayerTestCommon,
                         combine({{10, 4, 8, 2}},         // input shapes
                                 {{0, 1, 2}, {1, 2, 0}},  // axes
                                 {{}, {3, 10, 8}}),       // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_negative_reversed_axes, DftLayerTestCommon,
                         combine({{10, 4, 8, 2}},    // input shapes
                                 {{-1, -2, -3}},     // axes
                                 {{}, {8, 10, 3}}),  // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_4d_single_axis, DftLayerTestCommon,
                         combine({{10, 4, 8, 2}},        // input shapes
                                 {{0}, {1}, {2}},        // axes
                                 {{}, {1}, {5}, {20}}),  // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DFT_5d, DftLayerTestCommon,
                         combine({{10, 4, 8, 2, 2}},    // input shapes
                                 {{0, 1, 2, 3}},        // axes
                                 {{}, {3, 10, 8, 6}}),  // signal sizes
                         DFTLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DFT_5d_tile, DftLayerTestCommon,
                         combine({{1, 80, 64, 64, 2}},  // input shapes
                                 {{2, 3}},              // axes
                                 {{}}),                 // signal sizes
                         DFTLayerTest::getTestCaseName);

}  // namespace
