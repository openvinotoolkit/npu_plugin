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
INSTANTIATE_TEST_CASE_P(precommit, KmbMVNCustomlayerTests, testing::ValuesIn(convertParams));
#endif
