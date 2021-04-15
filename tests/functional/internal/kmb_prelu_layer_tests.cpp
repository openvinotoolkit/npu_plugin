//
// Copyright 2021 Intel Corporation.
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

INSTANTIATE_TEST_CASE_P(precommit, KmbPReluLayerTests,
    testing::Combine(testing::ValuesIn(PReluParams), testing::ValuesIn(layerType)));
