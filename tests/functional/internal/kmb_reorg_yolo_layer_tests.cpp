//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

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

    const auto tolerance = 0.05f;

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
        IE_ASSERT(data != nullptr);
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

#ifdef KMB_HAS_CUSTOM_OCL_KERNELS
INSTANTIATE_TEST_SUITE_P(precommit, KmbReorgYoloLayerTests,
    testing::Combine(
        testing::ValuesIn(reorgYoloParams),
        testing::Values<Layout>(NCHW, NHWC),
        testing::Values<UseCustomLayers>(KernelType::Ocl)));
#endif
