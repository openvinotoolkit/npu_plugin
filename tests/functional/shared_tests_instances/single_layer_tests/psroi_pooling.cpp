// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/psroi_pooling.hpp"

// #include <vector>

// #include "kmb_layer_test.hpp"
// // #include <common/functions.h>
// #include "common_test_utils/test_constants.hpp"


// namespace LayerTestsDefinitions {

// class KmbPSROIPoolingLayerTest : public PSRoiPoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
//     void SkipBeforeLoad() override {
//         std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationParam;
//         std::tie(activationParam,
//                  std::ignore, std::ignore, std::ignore, std::ignore,
//                  std::ignore, std::ignore, std::ignore) = GetParam();

//         const auto activationType = activationParam.first;

//         if (isCompilerMCM()) {
//             if (supportedTypesMCM.find(activationType) == supportedTypesMCM.end()) {
//                 throw LayerTestsUtils::KmbSkipTestException("Unsupported activation types in MCM compiler");
//             }
//         } else {
//             if (supportedTypesMLIR.find(activationType) == supportedTypesMLIR.end()) {
//                 throw LayerTestsUtils::KmbSkipTestException("Experimental compiler doesn't supports activation type " +
//                                                             LayerTestsDefinitions::activationNames[activationType] +
//                                                             " yet");
//             }
//         }

//         // [Track number: #E20853]
//         if (getBackendName(*getCore()) == "LEVEL0") {
//                 throw LayerTestsUtils::KmbSkipTestException("Level0: sporadic failures on device");
//             }
//     }
// };
// }


// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/psroi_pooling.hpp"

namespace LayerTestsDefinitions {

class KmbPSROIPoolingLayerTest : public PSROIPoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            if (envConfig.IE_KMB_TESTS_RUN_INFER) {
                // [Track number: S#44493]
                // Test hangs on the the board
                throw LayerTestsUtils::KmbSkipTestException("Issues with MCM compiler");
            }
        }
    }
};

TEST_P(KmbPSROIPoolingLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const auto genParams(const  std::vector<std::vector<size_t>> inShapes,const  std::vector<std::vector<size_t>> coordShapes, const size_t output_dim, 
                     const size_t group_size,const float spatial_scale, const size_t spatial_bins_x, const size_t spatial_bins_y,
                     const std::string mode)
{
    const std::vector<InferenceEngine::Precision> netPRCs = {InferenceEngine::Precision::FP16,
                                                             InferenceEngine::Precision::FP32};
    return ::testing::Combine(
        ::testing::ValuesIn(inShapes),     ::testing::ValuesIn(coordShapes), ::testing::Values(output_dim),
        ::testing::Values(group_size),     ::testing::Values(spatial_scale), ::testing::Values(spatial_bins_x),
        ::testing::Values(spatial_bins_y), ::testing::Values(mode),          ::testing::ValuesIn(netPRCs),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));
}

#define GEN_TEST(no,inShapes,coordShapes,output_dim,group_size,spatial_scale,spatial_bins_x,spatial_bins_y,mode)\
INSTANTIATE_TEST_CASE_P( \
        smoke_TestsPSROIPooling_Case ## no, \
        KmbPSROIPoolingLayerTest, \
        genParams(inShapes,coordShapes,output_dim,group_size,spatial_scale,spatial_bins_x,spatial_bins_y,mode),\
        KmbPSROIPoolingLayerTest::getTestCaseName)

GEN_TEST( 0,
          (std::vector<std::vector<size_t>>{{1, 392, 14, 14}}), //inShapes
          (std::vector<std::vector<size_t>>{{300, 5}}), //coordShapes
          8u,       //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 1,
          (std::vector<std::vector<size_t>>{{1, 1029, 14, 14}}), //inShapes
          (std::vector<std::vector<size_t>>{{300, 5}}), //coordShapes
          21u,      //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 2,
          (std::vector<std::vector<size_t>>{{1, 392, 38, 64}}), //inShapes
          (std::vector<std::vector<size_t>>{{300, 5}}), //coordShapes
          8u,       //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 3,
          (std::vector<std::vector<size_t>>{{1, 1029, 38, 64}}), //inShapes
          (std::vector<std::vector<size_t>>{{300, 5}}), //coordShapes
          21u,      //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 4,
          (std::vector<std::vector<size_t>>{{1, 98, 34, 62}}), //inShapes
          (std::vector<std::vector<size_t>>{{200, 5}}), //coordShapes
          2u,       //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 5,
          (std::vector<std::vector<size_t>>{{1, 392, 34, 62}}), //inShapes
          (std::vector<std::vector<size_t>>{{200, 5}}), //coordShapes
          8u,       //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 6,
          (std::vector<std::vector<size_t>>{{1, 49 * 5, 45, 80}}), //inShapes
          (std::vector<std::vector<size_t>>{{300, 5}}), //coordShapes
          5u,       //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 7,
          (std::vector<std::vector<size_t>>{{1, 7 * 7 * 8, 14, 14}}), //inShapes
          (std::vector<std::vector<size_t>>{{5, 5}}), //coordShapes
          8u,       //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);
GEN_TEST( 8,
          (std::vector<std::vector<size_t>>{{1, 49 * 1, 14, 14}}), //inShapes
          (std::vector<std::vector<size_t>>{{1, 5}}), //coordShapes
          1u,       //output_dim
          7u,       //group_size
          0.0625f,  //spatial_scale
          1u,       //spatial_bins_x
          1u,       //spatial_bins_y
          "average" //mode
);