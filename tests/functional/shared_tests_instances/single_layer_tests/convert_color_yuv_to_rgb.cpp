//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/convert_color_i420.hpp"
#include "single_layer_tests/convert_color_nv12.hpp"

namespace LayerTestsDefinitions {

class VPUXConvertColorNV12LayerTest :
        public ConvertColorNV12LayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXConvertColorI420LayerTest :
        public ConvertColorI420LayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class VPUXConvertColorNV12LayerTest_VPU3700 : public VPUXConvertColorNV12LayerTest {};
class VPUXConvertColorI420LayerTest_VPU3700 : public VPUXConvertColorI420LayerTest {};

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

class VPUXConvertColorNV12LayerTest_VPU3720 : public VPUXConvertColorNV12LayerTest {};
class VPUXConvertColorI420LayerTest_VPU3720 : public VPUXConvertColorI420LayerTest {};

TEST_P(VPUXConvertColorNV12LayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXConvertColorI420LayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// N,H,W,C
ov::Shape inShapes[] = {{1, 4, 8, 1}, {1, 64, 32, 1}, {3, 128, 128, 1}};

ov::element::Type dTypes[] = {
        ov::element::f16,
};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorNV12, VPUXConvertColorNV12LayerTest_VPU3700,
                         testing::Combine(testing::ValuesIn(inShapes),
                                          testing::ValuesIn(dTypes),     // elem Type
                                          testing::Values(true, false),  // conv_to_RGB
                                          testing::Values(true, false),  // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorNV12LayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorI420, VPUXConvertColorI420LayerTest_VPU3700,
                         testing::Combine(testing::ValuesIn(inShapes),
                                          testing::ValuesIn(dTypes),     // elem Type
                                          testing::Values(true, false),  // conv_to_RGB
                                          testing::Values(true, false),  // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorNV12LayerTest_VPU3700::getTestCaseName);

// VPU3720
INSTANTIATE_TEST_SUITE_P(smoke_precommit_ConvertColorNV12_VPU3720, VPUXConvertColorNV12LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 4, 8, 1}),  // non-QVGA
                                          testing::Values(ov::element::f16),       // elem Type
                                          testing::Values(true, false),            // conv_to_RGB
                                          testing::Values(true, false),            // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorNV12LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_VPU3720, VPUXConvertColorNV12LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 240, 320, 1}),  // QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorNV12LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_VPU3720_tiling, VPUXConvertColorNV12LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 662, 982, 1}),  // non-QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorNV12LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ConvertColorI420_VPU3720, VPUXConvertColorI420LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 4, 8, 1}),  // non-QVGA
                                          testing::Values(ov::element::f16),       // elem Type
                                          testing::Values(true, false),            // conv_to_RGB
                                          testing::Values(true, false),            // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorI420LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_VPU3720, VPUXConvertColorI420LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 240, 320, 1}),  // QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorI420LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_VPU3720_tiling, VPUXConvertColorI420LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 662, 982, 1}),  // non-QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXConvertColorI420LayerTest_VPU3720::getTestCaseName);
}  // namespace
