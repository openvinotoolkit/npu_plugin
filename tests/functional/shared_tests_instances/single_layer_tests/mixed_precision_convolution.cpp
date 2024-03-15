// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "shared_test_classes/layer/mixed_precision_convolution.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace LayerTestsDefinitions {

class MixedPrecisionConvLayerTestCommon : public MixedPrecisionConvLayerTest {};

using MixedPrecisionConvLayerTest_NPU3700 = MixedPrecisionConvLayerTestCommon;
using MixedPrecisionConvLayerTest_NPU3720 = MixedPrecisionConvLayerTestCommon;

TEST_P(MixedPrecisionConvLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(MixedPrecisionConvLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

}  // namespace LayerTestsDefinitions

using namespace InferenceEngine;
using namespace LayerTestsDefinitions;

namespace {

/* ============= NPU3720 ============= */

const auto conv2DParams = ::testing::Combine(::testing::ValuesIn<SizeVector>({{1, 1}}),              // kernels
                                             ::testing::ValuesIn<SizeVector>({{1, 1}}),              // strides
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                                             ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                                             ::testing::Values(16),                                  // numOutChannels
                                             ::testing::Values(255),                                 // quantLevels
                                             ::testing::Values(ngraph::helpers::Pertensor)           // quantGranularity
);

INSTANTIATE_TEST_CASE_P(smoke_precommit_mixed_precision_Convolution2D, MixedPrecisionConvLayerTest_NPU3700,
                        ::testing::Combine(conv2DParams,
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 16, 16}}),  // inputShapes
                                           ::testing::Values(ov::test::utils::DEVICE_NPU)),     // targetDevice
                        MixedPrecisionConvLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_mixed_precision_Convolution2D, MixedPrecisionConvLayerTest_NPU3720,
                        ::testing::Combine(conv2DParams,
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 16, 16}}),  // inputShapes
                                           ::testing::Values(ov::test::utils::DEVICE_NPU)),     // targetDevice
                        MixedPrecisionConvLayerTest_NPU3720::getTestCaseName);

}  // namespace
