// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/grn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbGRNLayerTest: public GrnLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbGRNLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::SizeVector> inShapes = {
    InferenceEngine::SizeVector {1, 3, 30, 30},
    InferenceEngine::SizeVector {1, 24, 128, 224},
};

const std::vector<float> biases = {
    0.33f, 
    1.1f,    
};

const auto params = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::NCHW),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inShapes),
    testing::ValuesIn(biases),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
    smoke_GRN_test,
    KmbGRNLayerTest,
    params,
    GrnLayerTest::getTestCaseName
);

}
