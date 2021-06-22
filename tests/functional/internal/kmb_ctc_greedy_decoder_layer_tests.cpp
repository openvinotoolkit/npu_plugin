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

    const auto userInDesc = TensorDesc(Precision::FP16, dims, layout);
    const auto seqIndDesc = TensorDesc(Precision::FP16, {dims[0], dims[1]}, Layout::NC);
    const auto userOutDesc = TensorDesc(Precision::FP32, {dims[1], dims[0], 1, 1}, Layout::NCHW);

    const auto tolerance = 0.f;

    const auto inputRange = std::make_pair(0, 4);

    registerBlobGenerator("input0", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    registerBlobGenerator("input1", seqIndDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.0f);
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

// [Track number: S#43632]
// Custom layers are sensitive to the network layout.
// MCM uses 'NHWC' layout by default ('NCHW' layout can be enabled for tests).
// CTCGreedyDecoder-1 has 'CHW' layout for the input.
// MCM has 'NHWC' layout instead.
// CHW - NHWC layout missmatch.
const std::vector<UseCustomLayers> layerType = {
// #ifdef KMB_HAS_CUSTOM_OCL_KERNELS
//    KernelType::Ocl,
// #endif
//#ifdef KMB_HAS_CUSTOM_CPP_KERNELS
//    KernelType::Cpp,
//#endif
    KernelType::Native
};

const std::vector<CTCDecoderTestParams> ctcParams = {
    CTCDecoderTestParams()
            .dims({88, 1, 71})
            .layout(Layout::CHW)
};

// [Track number: #-12236]
INSTANTIATE_TEST_SUITE_P(DISABLED_precommit, KmbCTCDecoderLayerTests,
    testing::Combine(testing::ValuesIn(ctcParams), testing::ValuesIn(layerType)));
