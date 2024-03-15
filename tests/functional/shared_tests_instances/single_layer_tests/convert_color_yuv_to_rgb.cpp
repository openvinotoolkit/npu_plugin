//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convert_color_i420.hpp"
#include "single_layer_tests/convert_color_nv12.hpp"
#include "vpu_ov1_layer_test.hpp"
#include "vpux_private_properties.hpp"

namespace LayerTestsDefinitions {

class ConvertColorNV12LayerTestCommon :
        public ConvertColorNV12LayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class ConvertColorI420LayerTestCommon :
        public ConvertColorI420LayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class ConvertColorNV12LayerTest_NPU3700 : public ConvertColorNV12LayerTestCommon {};
class ConvertColorI420LayerTest_NPU3700 : public ConvertColorI420LayerTestCommon {};

class ConvertColorNV12LayerTest_NPU3720 : public ConvertColorNV12LayerTestCommon {};
class ConvertColorI420LayerTest_NPU3720 : public ConvertColorI420LayerTestCommon {};

// NPU3700
TEST_P(ConvertColorNV12LayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ConvertColorI420LayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

// NPU3720
TEST_P(ConvertColorNV12LayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(ConvertColorI420LayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// N,H,W,C
ov::Shape inShapesNPU3700[] = {{1, 4, 8, 1}, {1, 64, 32, 1}, {3, 128, 128, 1}};
ov::Shape inShapes[] = {{1, 240, 320, 1}, {1, 4, 8, 1}, {1, 662, 982, 1}, {3, 128, 128, 1}};

ov::element::Type dTypes[] = {
        ov::element::f16,
};

const auto paramsNPU3700 = testing::Combine(testing::ValuesIn(inShapesNPU3700),
                                            testing::ValuesIn(dTypes),     // elem Type
                                            testing::Values(true, false),  // conv_to_RGB
                                            testing::Values(true, false),  // is_single_plane
                                            testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto params = testing::Combine(testing::ValuesIn(inShapes),
                                     testing::ValuesIn(dTypes),     // elem Type
                                     testing::Values(true, false),  // conv_to_RGB
                                     testing::Values(true, false),  // is_single_plane
                                     testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
// NPU3700
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12, ConvertColorNV12LayerTest_NPU3700, paramsNPU3700,
                         ConvertColorNV12LayerTest_NPU3700::getTestCaseName);

// [Tracking number: E#93409]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorI420, ConvertColorI420LayerTest_NPU3700, paramsNPU3700,
                         ConvertColorI420LayerTest_NPU3700::getTestCaseName);

// NPU3720
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12, ConvertColorNV12LayerTest_NPU3720, params,
                         ConvertColorNV12LayerTest_NPU3720::getTestCaseName);

// [Tracking number: E#93409]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorI420, ConvertColorNV12LayerTest_NPU3720, params,
                         ConvertColorNV12LayerTest_NPU3720::getTestCaseName);

}  // namespace
