//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convert_color_i420.hpp"
#include "single_layer_tests/convert_color_nv12.hpp"
#include "vpu_ov1_layer_test.hpp"
#include "vpux_private_config.hpp"

namespace LayerTestsDefinitions {

class VPUXConvertColorNV12LayerTest :
        public ConvertColorNV12LayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class VPUXConvertColorI420LayerTest :
        public ConvertColorI420LayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VPUXConvertColorNV12LayerTest_VPU3700 : public VPUXConvertColorNV12LayerTest {};
class VPUXConvertColorI420LayerTest_VPU3700 : public VPUXConvertColorI420LayerTest {};

class VPUXConvertColorNV12LayerTest_VPU3720 : public VPUXConvertColorNV12LayerTest {};
class VPUXConvertColorI420LayerTest_VPU3720 : public VPUXConvertColorI420LayerTest {};

// VPU3700
TEST_P(VPUXConvertColorNV12LayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXConvertColorI420LayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

// VPU3720
TEST_P(VPUXConvertColorNV12LayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXConvertColorI420LayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// N,H,W,C
ov::Shape inShapesVPU3700[] = {{1, 4, 8, 1}, {1, 64, 32, 1}, {3, 128, 128, 1}};
ov::Shape inShapesVPUX[] = {{1, 240, 320, 1}, {1, 4, 8, 1}, {1, 662, 982, 1}, {3, 128, 128, 1}};

ov::element::Type dTypes[] = {
        ov::element::f16,
};

const auto paramsVPU3700 = testing::Combine(testing::ValuesIn(inShapesVPU3700),
                                            testing::ValuesIn(dTypes),     // elem Type
                                            testing::Values(true, false),  // conv_to_RGB
                                            testing::Values(true, false),  // is_single_plane
                                            testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto paramsVPUX = testing::Combine(testing::ValuesIn(inShapesVPUX),
                                         testing::ValuesIn(dTypes),     // elem Type
                                         testing::Values(true, false),  // conv_to_RGB
                                         testing::Values(true, false),  // is_single_plane
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// VPU3700
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorNV12, VPUXConvertColorNV12LayerTest_VPU3700, paramsVPU3700,
                         VPUXConvertColorNV12LayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorI420, VPUXConvertColorI420LayerTest_VPU3700, paramsVPU3700,
                         VPUXConvertColorNV12LayerTest_VPU3700::getTestCaseName);

// VPU3720
INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_VPU3720, VPUXConvertColorNV12LayerTest_VPU3720, paramsVPUX,
                         VPUXConvertColorNV12LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_VPU3720, VPUXConvertColorI420LayerTest_VPU3720, paramsVPUX,
                         VPUXConvertColorI420LayerTest_VPU3720::getTestCaseName);

}  // namespace
