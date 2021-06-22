//
// Copyright 2021 Intel Corporation.
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

struct PReluTestParams final {
    PReluParams params;

    PARAMETER(SizeVector, dims);
    PARAMETER(Layout, layout);
};

std::ostream& operator<<(std::ostream& os, const PReluTestParams& p) {
    vpu::formatPrint(os, "dims: %l, layout: %l", p.dims(), p.layout());
    return os;
}

class KmbPReluLayerTests :
    public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<PReluTestParams, UseCustomLayers>> {};

TEST_P(KmbPReluLayerTests, accuracy) {
    
    const auto& p = std::get<0>(GetParam());
    const auto& useCustomLayers = std::get<1>(GetParam());

    const auto dims = p.dims();
    const auto layout = p.layout();

    const auto userInDesc = TensorDesc(Precision::FP16, dims, layout);
    const auto seqIndDesc = TensorDesc(Precision::FP16, {dims[1]}, Layout::C);
    const auto userOutDesc = TensorDesc(Precision::FP32, dims, layout);

    const auto tolerance = 0.005f;

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, -10.f, 10.f);
    });

    registerBlobGenerator("weights", seqIndDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, -1.f, -0.1f);
    });

    const auto netBuilder = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<PReluLayerDef>("PRelu", p.params)
                .input("input")
                .weights(getBlobByName("weights"))
                .build()
            .addNetOutput(PortInfo("PRelu"))
            .setUserOutput(PortInfo("PRelu"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(useCustomLayers)
            .finalize();
    };

    runTest(netBuilder, tolerance, CompareMethod::Absolute);
}

const std::vector<PReluTestParams> PReluParams = {
        PReluTestParams()
            .dims({1, 3, 128, 224})
            .layout(Layout::NCHW)
};

const std::vector<UseCustomLayers> layerType = {
    KernelType::Native
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbPReluLayerTests,
    testing::Combine(testing::ValuesIn(PReluParams), testing::ValuesIn(layerType)));
