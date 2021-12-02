// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include <kmb_layer_test.hpp>
#include "ngraph_functions/preprocess/preprocess_builders.hpp"


inline std::vector<ov::builder::preprocess::preprocess_func> nv12_convert_preprocess_functions() {
    return std::vector<ov::builder::preprocess::preprocess_func> {
            ov::builder::preprocess::preprocess_func(ov::builder::preprocess::cvt_color_nv12_to_rgb_single_plane, "cvt_color_nv12_to_rgb_single_plane", 1.f),
            ov::builder::preprocess::preprocess_func(ov::builder::preprocess::cvt_color_nv12_to_bgr_two_planes, "cvt_color_nv12_to_bgr_two_planes", 1.f),
            ov::builder::preprocess::preprocess_func(ov::builder::preprocess::cvt_color_nv12_cvt_layout_resize, "cvt_color_nv12_cvt_layout_resize", 1.f)
    };
}


using namespace SubgraphTestsDefinitions;

class VPUXPreProcessCompileTest : virtual public PrePostProcessTest,
                                  virtual public LayerTestsUtils::KmbLayerTestsCommon {
public:
    void SetUp() override {
        SkipBeforeInfer();
        PrePostProcessTest::SetUp();
    }
protected:
    std::map<std::string, std::string> config;
};

TEST_P(VPUXPreProcessCompileTest, CompareWithRefs) {
    KmbLayerTestsCommon::useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    KmbLayerTestsCommon::Run();
}


// [Track number: S#69189]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_PrePostProcess, PrePostProcessTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(ov::builder::preprocess::generic_preprocess_functions()),
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                         PrePostProcessTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_NV12ConvertCompilePreProcess, VPUXPreProcessCompileTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(nv12_convert_preprocess_functions()),
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                         PrePostProcessTest::getTestCaseName);
