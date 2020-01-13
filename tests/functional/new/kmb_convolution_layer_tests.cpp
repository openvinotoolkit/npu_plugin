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

#include "test_model/kmb_tests_base.hpp"

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

TEST_P(KmbConvolutionLayerTests, DISABLED_Single_FP16) {
    const auto& p = GetParam();

    const auto inDesc = TensorDesc(Precision::FP32, p._inDims, Layout::NCHW);

    const auto weights = genConvWeights(p._convParams, p._inDims.at(1), Precision::FP32, rd, -1.0f, 1.0f);
    const auto biases = genConvBiases(p._convParams, Precision::FP32, rd, -1.0f, 1.0f);

    TestNetwork testNet;
    testNet
        .setUserInput("input", inDesc.getPrecision(), inDesc.getLayout())
        .addNetInput("input", inDesc.getDims(), Precision::FP32)
        .addLayer<ConvolutionLayerDef>("conv", p._convParams)
            .input("input")
            .weights(weights)
            .biases(biases)
            .build()
        .addNetOutput(PortInfo("conv"))
        .finalize();

    const auto inputsGenerator = [&]() -> BlobMap {
        const auto input = genBlobUniform(inDesc, rd, -1.0f, 1.0f);
        return {{"input", input}};
    };

    runTest(testNet, inputsGenerator, 0.2f, CompareMethod::Absolute);
}

TEST_P(KmbConvolutionLayerTests, DISABLED_Single_FakeQuantize) {
    const auto& p = GetParam();

    const auto inDesc = TensorDesc(Precision::FP32, p._inDims, Layout::NCHW);

    const auto weights = genConvWeights(p._convParams, p._inDims.at(1), Precision::FP32, rd, -1.0f, 1.0f);
    const auto biases = genConvBiases(p._convParams, Precision::FP32, rd, -1.0f, 1.0f);

    TestNetwork testNet;
    testNet
        .setUserInput("input", inDesc.getPrecision(), inDesc.getLayout())
        .addNetInput("input", inDesc.getDims(), Precision::FP32)
        .addConst("weights", weights)
        .addLayer<FakeQuantizeLayerDef>("input_quantize", 256)
            .input("input")
            .low(-1.0f)
            .high(1.0f)
            .build()
        .addLayer<FakeQuantizeLayerDef>("weights_quantize", 256)
            .input("weights")
            .low(-1.0f)
            .high(1.0f)
            .build()
        .addLayer<ConvolutionLayerDef>("conv", p._convParams)
            .input("input_quantize")
            .weights("weights_quantize")
            .biases(biases)
            .build()
        .addNetOutput(PortInfo("conv"))
        .finalize();

    const auto inputsGenerator = [&]() -> BlobMap {
        const auto input = genBlobUniform(inDesc, rd, -1.0f, 1.0f);
        return {{"input", input}};
    };

    runTest(testNet, inputsGenerator, 0.0001f, CompareMethod::Absolute);
}

const std::vector<ConvTestParams> convParams {
    ConvTestParams()
        .inDims({1, 256, 56, 56})
        .convParams(ConvolutionParams().outChannels(128).kernel({1, 1}).strides({2, 2}).pad({0, 0, 0, 0}))
};

INSTANTIATE_TEST_CASE_P(SomeCase, KmbConvolutionLayerTests, testing::ValuesIn(convParams));
