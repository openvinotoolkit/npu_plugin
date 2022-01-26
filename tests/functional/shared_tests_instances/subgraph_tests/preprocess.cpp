// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/preprocess.hpp"
#include <vpux_layer_test.hpp>
#include "ngraph_functions/preprocess/preprocess_builders.hpp"


inline std::shared_ptr<ov::Model> create_preprocess_1input(ov::element::Type type,
                                                           const ov::PartialShape& shape) {
    auto data1 = std::make_shared<ov::op::v0::Parameter>(type, shape);
    data1->set_friendly_name("input1");
    data1->output(0).get_tensor().set_names({"input1"});
    std::shared_ptr<ov::op::v0::Result> res;
    auto op1 = std::make_shared<ov::op::v0::Relu>(data1);
    res = std::make_shared<ov::op::v0::Result>(op1);
    res->set_friendly_name("Result1");
    res->output(0).get_tensor().set_names({"Result1"});
    return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data1});
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_rgb_single_plane() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, {{1, 20, 20, 3}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_bgr_single_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, {{1, 20, 20, 3}});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_bgr_two_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_nv12_to_rgb_two_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::NV12_TWO_PLANES);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    function = p.build();
    return function;
}

inline std::shared_ptr<ov::Model> cvt_color_i420_to_rgb_single_plane() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::I420_SINGLE_PLANE);
    p.input().preprocess().convert_color(ColorFormat::RGB);
    return p.build();
}

inline std::shared_ptr<ov::Model> cvt_color_i420_to_bgr_three_planes() {
    using namespace ov::preprocess;
    auto function = create_preprocess_1input(ov::element::f32, ov::PartialShape{1, 20, 20, 3});
    auto p = PrePostProcessor(function);
    p.input().tensor().set_color_format(ColorFormat::I420_THREE_PLANES);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    return p.build();
}

inline std::vector<ov::builder::preprocess::preprocess_func> nv12_convert_preprocess_functions() {
    return std::vector<ov::builder::preprocess::preprocess_func> {
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_rgb_single_plane, "cvt_color_nv12_to_rgb_single_plane", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_bgr_single_planes, "cvt_color_nv12_to_bgr_single_planes", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_bgr_two_planes, "cvt_color_nv12_to_bgr_two_planes", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_nv12_to_rgb_two_planes, "cvt_color_nv12_to_rgb_two_planes", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_i420_to_rgb_single_plane, "cvt_color_i420_to_rgb_single_plane", 1.f),
            ov::builder::preprocess::preprocess_func(cvt_color_i420_to_bgr_three_planes, "cvt_color_i420_to_bgr_three_planes", 1.f),
    };
}


using namespace SubgraphTestsDefinitions;

class VPUXPreProcessTest : virtual public PrePostProcessTest,
                                  virtual public VPUXLayerTestsUtils::VPUXLayerTestsCommon {
public:
    void SetUp() override {
        PrePostProcessTest::SetUp();
    }
protected:
    std::map<std::string, std::string> config;
};

TEST_P(VPUXPreProcessTest, CompareWithRefs) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    run();
}


INSTANTIATE_TEST_SUITE_P(smoke_PrePostProcess, VPUXPreProcessTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(nv12_convert_preprocess_functions()),
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                         VPUXPreProcessTest::getTestCaseName);
