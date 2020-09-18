//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <common/include/vpu/utils/ie_helpers.hpp>
#include <single_layer_common.hpp>

#include "test_model/kmb_test_base.hpp"

struct RegionYoloTestParams final {
    RegionYoloParams params;

    LAYER_PARAMETER(size_t, classes);
    LAYER_PARAMETER(size_t, coords);
    LAYER_PARAMETER(size_t, regions);
    LAYER_PARAMETER(std::vector<int64_t>, mask);
    LAYER_PARAMETER(bool, doSoftmax);

    PARAMETER(size_t, width);
    PARAMETER(size_t, height);
};

std::ostream& operator<<(std::ostream& os, const RegionYoloTestParams& p) {
    const auto num = p.doSoftmax() ? p.regions() : p.mask().size();
    const auto& dims = SizeVector{1,
                                    num * (p.coords() + p.classes() + 1), p.height(), p.width()};

    vpu::formatPrint(os, "dims: %v, classes: %l, coords: %l, regions: %l, maskSize: %l, doSoftmax: %l",
        dims, p.classes(), p.coords(), p.regions(), p.mask().size(), p.doSoftmax());
    return os;
}

PRETTY_PARAM(DoSoftmax, bool)

class KmbRegionYoloLayerTests :
    public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<RegionYoloTestParams, DoSoftmax, Layout, UseCustomLayers>> {};

TEST_P(KmbRegionYoloLayerTests, accuracy) {
    auto p = std::get<0>(GetParam());
    p.params.doSoftmax = std::get<1>(GetParam());

    const auto& layout = std::get<2>(GetParam());
    const auto& useCustomLayer = std::get<3>(GetParam());

    const auto tolerance = 0.05f;

    const auto num = p.doSoftmax() ? p.regions() : p.mask().size();
    const auto& inDims = SizeVector{1, num * (p.coords() + p.classes() + 1), 13, 13};

    const auto userInDesc = TensorDesc(Precision::FP16, inDims, layout);
    const auto userOutDesc = TensorDesc(Precision::FP32, inDims, layout);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet.setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<RegionYoloLayerDef>("RegionYolo", p.params)
            .input("input")
            .build()
            .addNetOutput(PortInfo("RegionYolo"))
            .setUserOutput(PortInfo("RegionYolo"), userOutDesc.getPrecision(), Layout::NCHW)
            .useCustomLayers(useCustomLayer)
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<RegionYoloTestParams> RegionYoloParams = {
    RegionYoloTestParams()
        .width(13)
        .height(13)
        .classes(20)
        .coords(4)
        .regions(5)
        .mask({0, 1, 2})
    };

INSTANTIATE_TEST_CASE_P(precommit, KmbRegionYoloLayerTests,
    testing::Combine(
        testing::ValuesIn(RegionYoloParams),
        testing::Values<DoSoftmax>(true, false),
        testing::Values<Layout>(Layout::NHWC, Layout::NCHW),
        testing::Values<UseCustomLayers>(false, true)));
