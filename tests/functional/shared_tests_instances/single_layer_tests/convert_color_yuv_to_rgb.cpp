//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/convert_color_i420.hpp"
#include "single_layer_tests/convert_color_nv12.hpp"

namespace LayerTestsDefinitions {

class KmbConvertColorNV12LayerTest :
        public ConvertColorNV12LayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertColorNV12LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbConvertColorI420LayerTest :
        public ConvertColorI420LayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertColorI420LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbConvertColorNV12LayerTest_VPU3720 : public KmbConvertColorNV12LayerTest {};

TEST_P(KmbConvertColorNV12LayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

class KmbConvertColorI420LayerTest_VPU3720 :
        public ConvertColorI420LayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertColorI420LayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
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

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorNV12, KmbConvertColorNV12LayerTest,
                         testing::Combine(testing::ValuesIn(inShapes),
                                          testing::ValuesIn(dTypes),     // elem Type
                                          testing::Values(true, false),  // conv_to_RGB
                                          testing::Values(true, false),  // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorNV12LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ConvertColorI420, KmbConvertColorI420LayerTest,
                         testing::Combine(testing::ValuesIn(inShapes),
                                          testing::ValuesIn(dTypes),     // elem Type
                                          testing::Values(true, false),  // conv_to_RGB
                                          testing::Values(true, false),  // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorNV12LayerTest::getTestCaseName);

// VPU3720
INSTANTIATE_TEST_SUITE_P(smoke_precommit_ConvertColorNV12_VPU3720, KmbConvertColorNV12LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 4, 8, 1}),  // QVGA
                                          testing::Values(ov::element::f16),       // elem Type
                                          testing::Values(true, false),            // conv_to_RGB
                                          testing::Values(true, false),            // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorNV12LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_VPU3720, KmbConvertColorNV12LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 240, 320, 1}),  // QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorNV12LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorNV12_VPU3720_tiling, KmbConvertColorNV12LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 662, 982, 1}),  // QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorNV12LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ConvertColorI420_VPU3720, KmbConvertColorI420LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 4, 8, 1}),  // QVGA
                                          testing::Values(ov::element::f16),       // elem Type
                                          testing::Values(true, false),            // conv_to_RGB
                                          testing::Values(true, false),            // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorI420LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_VPU3720, KmbConvertColorI420LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 240, 320, 1}),  // QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorI420LayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertColorI420_VPU3720_tiling, KmbConvertColorI420LayerTest_VPU3720,
                         testing::Combine(testing::Values(ov::Shape{1, 662, 982, 1}),  // QVGA
                                          testing::Values(ov::element::f16),           // elem Type
                                          testing::Values(true, false),                // conv_to_RGB
                                          testing::Values(true, false),                // is_single_plane
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbConvertColorI420LayerTest_VPU3720::getTestCaseName);
}  // namespace
