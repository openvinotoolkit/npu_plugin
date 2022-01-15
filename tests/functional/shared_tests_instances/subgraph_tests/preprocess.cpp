// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include <vpux_layer_test.hpp>
#include "ngraph_functions/preprocess/preprocess_builders.hpp"

inline std::shared_ptr<ov::Model> create_preprocess_1input(ov::element::Type type,
                                                          const std::vector<std::vector<size_t>> inputShape) {
    auto params = ngraph::builder::makeParams(type, inputShape);
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, 0);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    return std::make_shared<ngraph::Function>(results, params, "concat");
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_rgb_single_plane() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::u8, {{1, 3, 128, 128}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_bgr_single_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::u8, {{1, 3, 128, 128}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    function = p.build();
    return function;
}

inline std::vector<ov::builder::preprocess::preprocess_func> nv12_convert_preprocess_functions() {
    return std::vector<ov::builder::preprocess::preprocess_func> {
           ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_rgb_single_plane, "cvt_color_nv12_to_rgb_single_plane", 1.f),
           ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_bgr_single_planes, "cvt_color_nv12_to_bgr_single_planes", 1.f)
    };
}

using namespace SubgraphTestsDefinitions;

class VPUXPreProcessCompileTest : virtual public PrePostProcessTest,
                                  virtual public VPUXLayerTestsUtils::VPUXLayerTestsCommon {
public:
    void SetUp() override {
        SkipBeforeInfer();
        PrePostProcessTest::SetUp();
    }
};

TEST_P(VPUXPreProcessCompileTest, CompareWithRefs) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    VPUXLayerTestsCommon::run();
}

INSTANTIATE_TEST_SUITE_P(smoke_NV12ConvertCompilePreProcess, VPUXPreProcessCompileTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(nv12_convert_preprocess_functions()),
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                         PrePostProcessTest::getTestCaseName);
