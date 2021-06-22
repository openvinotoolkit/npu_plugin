//
// Copyright 2019-2020 Intel Corporation.
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

using ConvTestParams = std::tuple<
    SizeVector,         // inDims
    ConvolutionParams,  // convParams
    bool,               // withBiases
    bool                // swMode
>;

void PrintTo(const ConvTestParams& p, std::ostream* os) {
    vpu::formatPrint(*os, "[inDims:%v, convParams:%v, withBiases:%v, swMode:%v]",
                     std::get<0>(p), std::get<1>(p), std::get<2>(p), std::get<3>(p));
}

class KmbConvolutionLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<ConvTestParams> {
public:
    const Precision netPresicion = Precision::FP32;

    TensorDesc userInDesc;
    const TensorDesc userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);

    const float base_tolerance = 1e-2f;
    float tolerance = 0.0f;

    std::map<std::string, std::string> config;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(KmbLayerTestBase::SetUp());

        const auto& p = GetParam();
        const auto& inDims = std::get<0>(p);
        const auto& convParams = std::get<1>(p);
        const auto swMode = std::get<3>(p);

        userInDesc = TensorDesc(Precision::U8, inDims, Layout::NHWC);
        tolerance = base_tolerance * convParams._kernel.x * convParams._kernel.y * inDims.at(1);

        registerBlobGenerator(
            "input", userInDesc,
            [&](const TensorDesc& desc) {
                return genBlobUniform(desc, rd, 0, 5);
            }
        );
        registerBlobGenerator(
            "weights", getConvWeightsDesc(convParams, inDims.at(1), netPresicion),
            [&](const TensorDesc& desc) {
                return genBlobUniform(desc, rd, 0.0f, 1.0f);
            }
        );
        registerBlobGenerator(
            "biases", getConvBiasesDesc(convParams, netPresicion),
            [&](const TensorDesc& desc) {
                return genBlobUniform(desc, rd, -1.0f, 1.0f);
            }
        );

        if (swMode) {
            config["VPU_COMPILER_REFERENCE_MODE"] = CONFIG_VALUE(YES);
        }
    }
};

TEST_P(KmbConvolutionLayerTests, FP16) {
    const auto& p = GetParam();
    const auto& convParams = std::get<1>(p);
    const auto withBiases = std::get<2>(p);
    const auto swMode = std::get<3>(p);

    if (!swMode) {
        SKIP_ON("KMB", "HDDL2", "VPUX", "HW FP16 Convolution is not supported yet");
    }

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<ConvolutionLayerDef>("conv", convParams)
                .input("input")
                .weights(getBlobByName("weights"))
                .biases(withBiases ? getBlobByName("biases") : nullptr)
                .build()
            .addNetOutput(PortInfo("conv"))
            .setUserOutput(PortInfo("conv"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();

        testNet.setCompileConfig(config);
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

TEST_P(KmbConvolutionLayerTests, FQ) {
    const auto& p = GetParam();
    const auto& convParams = std::get<1>(p);
    const auto withBiases = std::get<2>(p);

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<FakeQuantizeLayerDef>("input_quantize", 256)
                .input("input")
                .low(0.0f, netPresicion)
                .high(255.0f, netPresicion)
                .build()
            .addLayer<FakeQuantizeLayerDef>("weights_quantize", 256)
                .input(getBlobByName("weights"))
                .low(0.0f, netPresicion)
                .high(1.0f, netPresicion)
                .build()
            .addLayer<ConvolutionLayerDef>("conv", convParams)
                .input("input_quantize")
                .weights("weights_quantize")
                .biases(withBiases ? getBlobByName("biases") : nullptr)
                .build()
            .addNetOutput(PortInfo("conv"))
            .setUserOutput(PortInfo("conv"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();

        testNet.setCompileConfig(config);
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<SizeVector> inDims = {
    {1, 3, 64, 64},
    {1, 32, 16, 16}
};

const std::vector<ConvolutionParams> simpleParams = {
    ConvolutionParams().outChannels(16).kernel(3).strides(1).pad(1).dilation(1),
    ConvolutionParams().outChannels(16).kernel(3).strides(2).pad(0).dilation(1),
};

const std::vector<ConvolutionParams> dilatedParams = {
    ConvolutionParams().outChannels(16).kernel(3).strides(1).pad(1).dilation(4),
    ConvolutionParams().outChannels(16).kernel(3).strides(2).pad(0).dilation(2),
};

INSTANTIATE_TEST_SUITE_P(
    precommit_Simple, KmbConvolutionLayerTests,
    testing::Combine(
        testing::ValuesIn(inDims),
        testing::ValuesIn(simpleParams),
        testing::Bool(),        // withBiases
        testing::Values(true)   // swMode
    )
);

INSTANTIATE_TEST_SUITE_P(
    precommit_Dilated, KmbConvolutionLayerTests,
    testing::Combine(
        testing::ValuesIn(inDims),
        testing::ValuesIn(dilatedParams),
        testing::Bool(),        // withBiases
        testing::Values(true)   // swMode
    )
);
