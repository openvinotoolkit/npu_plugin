//
// Copyright 2019-2020 Intel Corporation.
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
        SKIP_ON("KMB", "HW FP16 Convolution is not supported yet");
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

// [Track number: S#26002]
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

INSTANTIATE_TEST_CASE_P(
    Simple, KmbConvolutionLayerTests,
    testing::Combine(
        testing::ValuesIn(inDims),
        testing::ValuesIn(simpleParams),
        testing::Bool(),        // withBiases
        testing::Values(true)   // swMode
    )
);

INSTANTIATE_TEST_CASE_P(
    Dilated, KmbConvolutionLayerTests,
    testing::Combine(
        testing::ValuesIn(inDims),
        testing::ValuesIn(dilatedParams),
        testing::Bool(),        // withBiases
        testing::Values(true)   // swMode
    )
);
