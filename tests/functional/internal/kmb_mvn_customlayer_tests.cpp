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

#include "kmb_mvn_test_params.hpp"
#include "test_model/kmb_test_base.hpp"

class KmbMVNCustomlayerTests : public KmbLayerTestBase, public testing::WithParamInterface<MVNTestParams> {};

TEST_P(KmbMVNCustomlayerTests, accuracy) {
    const auto &p = GetParam();

    const auto& dims = p.dims();
    const auto userInDesc = TensorDesc(Precision::FP16, dims, Layout::NCHW);
    const auto userOutDesc = TensorDesc(Precision::FP32, dims, Layout::NCHW);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<MVNLayerDef>("mvn", p.params)
                .input("input")
                .build()
            .addNetOutput(PortInfo("mvn"))
            .setUserOutput(PortInfo("mvn"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(KernelType::Ocl)
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<MVNTestParams> convertParams = {
        MVNTestParams()
            .dims({1, 3, 512, 896})
            .normalize_variance(true)
            .across_channels(true)
};

#ifdef KMB_HAS_CUSTOM_OCL_KERNELS
INSTANTIATE_TEST_SUITE_P(precommit, KmbMVNCustomlayerTests, testing::ValuesIn(convertParams));
#endif
