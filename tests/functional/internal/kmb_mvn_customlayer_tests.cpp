//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "kmb_mvn_test_params.hpp"
#include "test_model/kmb_test_base.hpp"

class KmbMVNCustomlayerTests : public KmbLayerTestBase, public testing::WithParamInterface<MVNTestParams> {};

// [Track number: E#20729]
TEST_P(KmbMVNCustomlayerTests, accuracy) {
    SKIP_ON("LEVEL0", "Sporadic failures on device");
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
INSTANTIATE_TEST_SUITE_P(DISABLED_precommit, KmbMVNCustomlayerTests, testing::ValuesIn(convertParams));
#endif
