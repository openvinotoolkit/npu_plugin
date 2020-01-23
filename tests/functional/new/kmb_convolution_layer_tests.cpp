//
// Copyright 2019 Intel Corporation.
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

struct ConvTestParams final {
    SizeVector _inDims;
    ConvolutionParams _convParams;

    ConvTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    ConvTestParams& convParams(const ConvolutionParams& convParams) {
        this->_convParams = convParams;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const ConvTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, convParams:%v]",
        p._inDims, p._convParams);
    return os;
}

class KmbConvolutionLayerTests : public KmbTestBase, public testing::WithParamInterface<ConvTestParams> {};

TEST_P(KmbConvolutionLayerTests, DISABLED_FP16) {
    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 3.0f);

    const auto weightsRange = std::make_pair(-1.0f, 1.0f);
    const auto biasesRange = std::make_pair(-1.0f, 1.0f);

    const auto tolerance = 1e-3f;  // obtained based on CPU plugin

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );
    registerBlobGenerator(
        "weights", getConvWeightsDesc(p._convParams, p._inDims.at(1), netPresicion),
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, weightsRange.first, weightsRange.second);
        }
    );
    registerBlobGenerator(
        "biases", getConvBiasesDesc(p._convParams, netPresicion),
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, biasesRange.first, biasesRange.second);
        }
    );

    TestNetwork testNet;
    testNet
        .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
        .addNetInput("input", userInDesc.getDims(), netPresicion)
        .addLayer<ConvolutionLayerDef>("conv", p._convParams)
            .input("input")
            .weights(getBlobByName("weights"))
            .biases(getBlobByName("biases"))
            .build()
        .addNetOutput(PortInfo("conv"))
        .setUserOutput(PortInfo("conv"), userOutDesc.getPrecision(), userOutDesc.getLayout())
        .finalize();

    runTest(testNet, tolerance, CompareMethod::Absolute);
}

TEST_P(KmbConvolutionLayerTests, DISABLED_FakeQuantize_ScaleShift) {
    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 255.0f);

    const auto inputScale = 1.0f / (inputRange.second - inputRange.first);
    const auto inputShift = 0.0f; // -0.5f;

    const auto weightsRange = std::make_pair(-1.0f, 1.0f);
    const auto biasesRange = std::make_pair(-1.0f, 1.0f);

    const auto tolerance = 1e-3f;  // obtained based on CPU plugin

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );
    registerBlobGenerator(
        "scales", TensorDesc(netPresicion, {1, p._inDims.at(1), 1, 1}, Layout::NCHW),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, inputScale);
        }
    );
    registerBlobGenerator(
        "shift", TensorDesc(netPresicion, {1, p._inDims.at(1), 1, 1}, Layout::NCHW),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, inputShift);
        }
    );
    registerBlobGenerator(
        "weights", getConvWeightsDesc(p._convParams, p._inDims.at(1), netPresicion),
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, weightsRange.first, weightsRange.second);
        }
    );
    registerBlobGenerator(
        "biases", getConvBiasesDesc(p._convParams, netPresicion),
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, biasesRange.first, biasesRange.second);
        }
    );

    TestNetwork testNet;
    testNet
        .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
        .addNetInput("input", userInDesc.getDims(), netPresicion)
        .addLayer<ScaleShiftLayerDef>("input_scale_shift")
            .input("input")
            .scale(getBlobByName("scales"))
            .shift(getBlobByName("shift"))
            .build()
        .addConst("weights", getBlobByName("weights"))
        .addLayer<FakeQuantizeLayerDef>("weights_quantize", 256)
            .input("weights")
            .low(weightsRange.first, netPresicion)
            .high(weightsRange.second, netPresicion)
            .build()
        .addLayer<ConvolutionLayerDef>("conv", p._convParams)
            .input("input_scale_shift")
            .weights("weights_quantize")
            .biases(getBlobByName("biases"))
            .build()
        .addNetOutput(PortInfo("conv"))
        .setUserOutput(PortInfo("conv"), userOutDesc.getPrecision(), userOutDesc.getLayout())
        .finalize();

    runTest(testNet, tolerance, CompareMethod::Absolute);
}

const std::vector<ConvTestParams> convParams {
    ConvTestParams()
        .inDims({1, 32, 8, 8})
        .convParams(ConvolutionParams().outChannels(16).kernel({1, 1}).strides({2, 2}).pad({0, 0, 0, 0}))
};

INSTANTIATE_TEST_CASE_P(SomeCase, KmbConvolutionLayerTests, testing::ValuesIn(convParams));
