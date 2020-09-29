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

struct GRNTestParams final {
    GRNParams params;

    LAYER_PARAMETER(float, bias);
    PARAMETER(SizeVector, dims);
    PARAMETER(Layout, layout);
};

std::ostream& operator<<(std::ostream& os, const GRNTestParams& p) {
    vpu::formatPrint(os, "dims: %l, layout: %l, bias: %l", p.dims(), p.layout(), p.bias());
    return os;
}

class KmbGRNLayerTests :
    public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<GRNTestParams, UseCustomLayers>> {};

TEST_P(KmbGRNLayerTests, accuracy) {
    const auto& p = std::get<0>(GetParam());
    const auto& useCustomLayers = std::get<1>(GetParam());

    const auto dims = p.dims();
    const auto layout = p.layout();

    const auto userInDesc = TensorDesc(Precision::FP16, dims, layout);
    const auto userOutDesc = TensorDesc(Precision::FP32, dims, layout);

    const auto tolerance = 1e-3f;

    const auto inputRange = std::make_pair(0.f, 10.f);

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    const auto netBuilder = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<GRNLayerDef>("GRN", p.params)
                .input("input")
                .build()
            .addNetOutput(PortInfo("GRN"))
            .setUserOutput(PortInfo("GRN"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(useCustomLayers)
            .finalize();
    };

    runTest(netBuilder, tolerance, CompareMethod::Absolute);
}

const std::vector<GRNTestParams> grnParams = {
        GRNTestParams()
            .dims({1, 24, 128, 224})
            .bias(1.0f)
            .layout(Layout::NCHW)
};

INSTANTIATE_TEST_CASE_P(precommit, KmbGRNLayerTests,
    testing::Combine(testing::ValuesIn(grnParams), testing::Values<UseCustomLayers>(true)));
