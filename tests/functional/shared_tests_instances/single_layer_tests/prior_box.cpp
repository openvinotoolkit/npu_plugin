// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/prior_box.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"


//tiny misspelling in 'prior_box.hpp'
using namespace LayerTestDefinitions;
//should have been 'LayerTestsDefinitions'

namespace LayerTestsDefinitions {

class KmbPriorBoxLayerTest: public PriorBoxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbPriorBoxLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions


using namespace LayerTestsDefinitions;

namespace {

const priorBoxSpecificParams param1 = { //(openvino eg)
    std::vector<float>{16.0},  // min_size
    std::vector<float>{38.46}, // max_size
    std::vector<float>{2.0},   // aspect_ratio
    std::vector<float>{},      // [density]
    std::vector<float>{},      // [fixed_ratio]
    std::vector<float>{},      // [fixed_size]
    false,                     // clip
    true,                      // flip
    16.0,                      // step
    0.5,                       // offset
    std::vector<float>{0.1, 0.1, 0.2, 0.2}, // variance
    false,                     // [scale_all_sizes]
    false                      // min_max_aspect_ratios_order ?
};

const priorBoxSpecificParams param2 = {
    std::vector<float>{2.0}, // min_size
    std::vector<float>{5.0}, // max_size
    std::vector<float>{1.5}, // aspect_ratio
    std::vector<float>{},    // [density]
    std::vector<float>{},    // [fixed_ratio]
    std::vector<float>{},    // [fixed_size]
    false,                   // clip
    false,                   // flip
    1.0,                     // step
    0.0,                     // offset
    std::vector<float>{},    // variance
    false,                   // [scale_all_sizes]
    false                    // min_max_aspect_ratios_order
};

const priorBoxSpecificParams param3 = {
    std::vector<float>{256.0}, // min_size
    std::vector<float>{315.0}, // max_size
    std::vector<float>{2.0}, // aspect_ratio
    std::vector<float>{},    // [density]
    std::vector<float>{},    // [fixed_ratio]
    std::vector<float>{},    // [fixed_size]
    true,                    // clip
    true,                    // flip
    1.0,                     // step
    0.0,                     // offset
    std::vector<float>{},    // variance
    true,                    // [scale_all_sizes]
    false                    // min_max_aspect_ratios_order
};

#define GEN_TEST(no,param,out_size,img_size)\
INSTANTIATE_TEST_CASE_P( \
        smoke_PriorBox_ ## no, \
        KmbPriorBoxLayerTest, \
        testing::Combine(  \
           testing::Values(param), \
           testing::Values(InferenceEngine::Precision::FP16), \
           testing::Values(InferenceEngine::Precision::UNSPECIFIED), \
           testing::Values(InferenceEngine::Precision::UNSPECIFIED), \
           testing::Values(InferenceEngine::Layout::ANY), \
           testing::Values(InferenceEngine::Layout::ANY), \
           testing::Values(out_size), \
           testing::Values(img_size), \
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)), \
        KmbPriorBoxLayerTest::getTestCaseName)

GEN_TEST(1, param1, (InferenceEngine::SizeVector{ 24, 42}), (InferenceEngine::SizeVector{348,672}) );
GEN_TEST(2, param2, (InferenceEngine::SizeVector{  2,  2}), (InferenceEngine::SizeVector{ 10, 10}) );
GEN_TEST(3, param3, (InferenceEngine::SizeVector{  1,  1}), (InferenceEngine::SizeVector{300,300}) );

}  // namespace
