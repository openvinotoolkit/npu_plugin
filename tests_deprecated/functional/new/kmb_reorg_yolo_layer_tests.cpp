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

struct ReorgYoloTestParams final {
    ReorgYoloParams params;

    LAYER_PARAMETER(int, stride);
    PARAMETER(SizeVector, dims);
};

std::ostream& operator<<(std::ostream& os, const ReorgYoloTestParams& p) {
    vpu::formatPrint(os, "dims: %v, stride: %t", p.dims(), p.stride());
    return os;
}

class KmbReorgYoloLayerTests :
    public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<ReorgYoloTestParams, Layout, UseCustomLayers>> {};

TEST_P(KmbReorgYoloLayerTests, accuracy) {
    const auto& p = std::get<0>(GetParam());
    const auto& layout = std::get<1>(GetParam());
    const auto& useCustomLayers = std::get<2>(GetParam());

    const auto tolerance = 0.0f;

    const auto stride = p.stride();

    const auto& inDims = p.dims();
    ASSERT_EQ(inDims[1] % (stride * stride), 0);

    const auto outDims = SizeVector{
        inDims[0],
        inDims[1] / (stride * stride),
        inDims[2] * stride,
        inDims[3] * stride};

    const auto userInDesc = TensorDesc(Precision::FP16, inDims, layout);
    const auto userOutDesc = TensorDesc(Precision::FP32, outDims, layout);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        auto blob = genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        auto data = blob->buffer().as<uint8_t *>();
        data[0] = 0;
        data[1] = 1;
        data[2] = 2;

        return blob;
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<ReorgYoloLayerDef>("ReorgYolo", p.params)
                .input("input")
                .build()
            .addNetOutput(PortInfo("ReorgYolo"))
            .setUserOutput(PortInfo("ReorgYolo"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(useCustomLayers)
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<ReorgYoloTestParams> reorgYoloParams = {
        ReorgYoloTestParams()
            .dims({1, 64, 26, 26})
            .stride(2)
};

INSTANTIATE_TEST_CASE_P(precommit, KmbReorgYoloLayerTests,
    testing::Combine(
        testing::ValuesIn(reorgYoloParams),
        testing::Values<Layout>(NCHW, NHWC),
        testing::Values<UseCustomLayers>(true)));
