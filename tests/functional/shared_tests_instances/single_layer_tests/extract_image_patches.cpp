// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/extract_image_patches.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbExtractImagePatchesTest : public ExtractImagePatchesTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbExtractImagePatchesTest, CompareWithRefs_MLIR) {
           useCompilerMLIR();
           Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

//TODO

}  // namespace
