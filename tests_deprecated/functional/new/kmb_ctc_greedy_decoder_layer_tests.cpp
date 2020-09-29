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

#include "test_model/kmb_test_base.hpp"

struct CTCDecoderTestParams final {
    CTCDecoderParams params;

    LAYER_PARAMETER(bool, mergeRepeated);
    PARAMETER(SizeVector, dims);
    PARAMETER(Layout, layout);
};

std::ostream& operator<<(std::ostream& os, const CTCDecoderTestParams& p) {
    vpu::formatPrint(os, "dims: %l, layout: %l, merge_repeated: %l", p.dims(), p.layout(), p.mergeRepeated());
    return os;
}

class KmbCTCDecoderLayerTests :
    public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<CTCDecoderTestParams, UseCustomLayers>> {};

TEST_P(KmbCTCDecoderLayerTests, accuracy) {
    const auto &p = std::get<0>(GetParam());
    const auto &useCustomLayers = std::get<1>(GetParam());

    const auto dims = p.dims();
    const auto layout = p.layout();

    const auto userInDesc = TensorDesc(Precision::U8, dims, layout);
    const auto seqIndDesc = TensorDesc(Precision::FP16, {1, 1, dims[2], dims[1]}, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP32, {1, 1, 1, dims[1]}, layout);

    const auto tolerance = 1e-3f;

    const auto inputRange = std::make_pair(0, 4);

    registerBlobGenerator("input0", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    registerBlobGenerator("input1", seqIndDesc, [&](const TensorDesc& desc) {
        return makeSingleValueBlob(desc, 1.0f);
    });

    const auto netBuilder = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input0", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input0", userInDesc.getDims(), Precision::FP32)
            .addLayer<CTCGreedyDecoderLayerDef>("CTCDecoder", p.params)
                .input0("input0")
                .input1(getBlobByName("input1"))
                .build()
            .addNetOutput(PortInfo("CTCDecoder"))
            .setUserOutput(PortInfo("CTCDecoder"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(useCustomLayers)
            .finalize();
    };

    runTest(netBuilder, tolerance, CompareMethod::Absolute);
}

const std::vector<CTCDecoderTestParams> ctcParams = {
    CTCDecoderTestParams()
            .dims({1, 88, 1, 71})
            .layout(Layout::NCHW)
};

INSTANTIATE_TEST_CASE_P(precommit, KmbCTCDecoderLayerTests,
    testing::Combine(testing::ValuesIn(ctcParams), testing::Values<UseCustomLayers>(true)));
