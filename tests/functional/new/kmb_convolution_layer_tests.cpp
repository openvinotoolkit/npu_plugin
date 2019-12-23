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

#include "kmb_tests_base.hpp"

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

TEST_P(KmbConvolutionLayerTests, Single_FakeQuantize_U8) {
    const auto& p = GetParam();

    const auto inDesc = TensorDesc(Precision::U8, p._inDims, Layout::NCHW);

    const auto weights = genConvWeights(p._convParams, p._inDims.at(1), Precision::FP32, rd, -1.0f, 1.0f);
    const auto biases = genConvBiases(p._convParams, Precision::FP32, rd, -1.0f, 1.0f);

    TestNetwork testNet;
    testNet
        .setUserInput("input", inDesc.getPrecision(), inDesc.getLayout())
        .addNetInput("input", inDesc.getDims(), Precision::FP32)
        .addConst("weights", weights)
        .addLayer<FakeQuantizeLayerDef>("input_quantize", 256)
            .input("input")
            .low(0.0f)
            .high(255.0f)
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
            .build();

    const auto inputsGenerator = [&]() -> BlobMap {
        const auto input = genBlobUniform(inDesc, rd, 0.0f, 255.0f);
        return {{"input", input}};
    };

    const auto inputs = getInputs(testNet, inputsGenerator);

    const auto convStats = [&]() {
        if (!RUN_REF_CODE) {
            return std::make_pair(0.0f, 1.0f);
        } else {
            std::cout << "=== COLLECT CONVOLUTION STATS" << std::endl;

            TestNetwork refNet = testNet;
            refNet
                .addNetOutput(PortInfo("conv"))
                .finalize();

            const auto refOutputs = refNet.calcRef(inputs);

            const auto& refOutput = refOutputs.at("conv");
            const auto refOutputPtr = refOutput->cbuffer().as<const float*>();
            IE_ASSERT(refOutputPtr != nullptr);

            const auto refOutputMinPtr = std::min_element(refOutputPtr, refOutputPtr + refOutput->size());
            const auto refOutputMaxPtr = std::max_element(refOutputPtr, refOutputPtr + refOutput->size());

            std::cout << "    Convolution stats: low:" << *refOutputMinPtr << " high:" << *refOutputMaxPtr << std::endl;

            return std::make_pair(*refOutputMinPtr, *refOutputMaxPtr);
        }
    }();

    testNet
        .addLayer<FakeQuantizeLayerDef>("conv_quantize", 256)
            .input("conv")
            .low(convStats.first)
            .high(convStats.second)
            .build()
        .addNetOutput(PortInfo("conv_quantize"))
        .setUserOutput(PortInfo("conv_quantize"), Precision::U8, Layout::NCHW)
        .finalize();

    auto exeNet = getExecNetwork(testNet);

    const auto refOutputs = getRefOutputs(testNet, inputs);

    if (RUN_INFER) {
        auto actualOutputs = runInfer(exeNet, inputs);

        if (DEVICE_NAME == "KMB") {
            // HACK: KMB plugin doesn't support dequantized FP32 output.
            IE_ASSERT(actualOutputs.size() == 1);
            auto& actualOutput = actualOutputs.begin()->second;
            actualOutput = dequantize(actualOutput, convStats.first, convStats.second, 256);
        }

        compareWithReference(actualOutputs, refOutputs, 0.0001f, CompareMethod::Absolute);
    }
}

const std::vector<ConvTestParams> convParams {
    ConvTestParams()
        .inDims({1, 256, 56, 56})
        .convParams(ConvolutionParams().outChannels(128).kernel({1, 1}).strides({2, 2}).pad({0, 0, 0, 0}))
};

INSTANTIATE_TEST_CASE_P(SomeCase, KmbConvolutionLayerTests, testing::ValuesIn(convParams));
