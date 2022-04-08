// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convert_color_nv12.hpp"
#include "single_layer_tests/convert_color_i420.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConvertColorNV12LayerTest: public ConvertColorNV12LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertColorNV12LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbConvertColorI420LayerTest: public ConvertColorI420LayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvertColorI420LayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// N,H,W,C
ov::Shape inShapes[] = {
 {1,   4,   8, 1},
 {1,  64,  32, 1},
 {3, 128, 128, 1}
};

ov::element::Type dTypes[] = {
  ov::element::f16,
};

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvertColorNV12,
        KmbConvertColorNV12LayerTest,
        testing::Combine(
           testing::ValuesIn(inShapes),
           testing::ValuesIn(dTypes),    // elem Type
           testing::Values(true, false), // conv_to_RGB
           testing::Values(true, false), // is_single_plane
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbConvertColorNV12LayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvertColorI420,
        KmbConvertColorI420LayerTest,
        testing::Combine(
           testing::ValuesIn(inShapes),
           testing::ValuesIn(dTypes),    // elem Type
           testing::Values(true, false), // conv_to_RGB
           testing::Values(true, false), // is_single_plane
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbConvertColorNV12LayerTest::getTestCaseName
);

}  // namespace
