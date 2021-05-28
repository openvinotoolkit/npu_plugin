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

    // Custom CPP layers fail
    // [Track number: E#11436]
    if (useCustomLayers == KernelType::Cpp)
    {
        SKIP_ON("KMB", "HDDL2", "VPUX", "Error in infer");
    }

#ifdef _WIN32
    // [Track number: E#13238]
    SKIP_ON("KMB", "HDDL2", "VPUX", "CallVpu error: -1");
#endif

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

const std::vector<UseCustomLayers> CustomLayersParams = {
#ifdef KMB_HAS_CUSTOM_OCL_KERNELS
    KernelType::Ocl,
#endif
#ifdef KMB_HAS_CUSTOM_CPP_KERNELS
    KernelType::Cpp
#endif
};

#if defined(KMB_HAS_CUSTOM_OCL_KERNELS) || defined(KMB_HAS_CUSTOM_CPP_KERNELS)
INSTANTIATE_TEST_CASE_P(precommit, KmbGRNLayerTests,
    testing::Combine(testing::ValuesIn(grnParams), testing::ValuesIn(CustomLayersParams)));
#endif
