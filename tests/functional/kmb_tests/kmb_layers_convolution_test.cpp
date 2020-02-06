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

#include <ie_common.h>
#include <ie_layers.h>

#include <cnn_network_int8_normalizer.hpp>
#include <conv_ref.hpp>
#include <ie_icnn_network_stats.hpp>
#include <ie_util_internal.hpp>
#include <kmb_conv_utils.hpp>
#include <pool_ref.hpp>
#include <vpu/kmb_plugin_config.hpp>

#include "common_test_utils/common_layers_params.hpp"
#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace ::testing;
using namespace InferenceEngine;
using namespace details;

struct convolution_test_params {
    SizeVector input_dim;
    CommonTestUtils::conv_common_params conv_params;
};

TBlob<uint8_t>::Ptr weightsBiasBlobPrepare(convolution_test_desc& convTestParam) {
    size_t weightsByteSize =
        getConvWeightsByteSize(convTestParam.input_dim, convTestParam.conv_params, convTestParam.conv_precision);
    size_t biasByteSize = getConvBiasesByteSize(convTestParam.conv_params, convTestParam.bias_precision);

    TBlob<uint8_t>::Ptr weightsBuffer = make_shared_blob<uint8_t>(
        {Precision::U8, {weightsByteSize + biasByteSize + convTestParam.weightsBufferOffset}, Layout::C});
    weightsBuffer->allocate();

    auto data = weightsBuffer->buffer().as<InferenceEngine::ie_fp16*>();
    fillRealBuffer<InferenceEngine::ie_fp16>(data + convTestParam.weightsBufferOffset / sizeof(ie_fp16),
        (weightsByteSize + biasByteSize) / sizeof(ie_fp16), PrecisionUtils::f32tof16(1.f),
        PrecisionUtils::f32tof16(1.f));

    return weightsBuffer;
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsConvolutionAfterScaleShift) {
    // TODO: tests fails. mcmCompiler compilation (Convolution with bias): Segmentation fault. Jira: VPUNND-1474
    const std::string model = conv_after_scale_shift;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsConvolutionAfterScaleShiftNoBias) {
    std::string model = conv_after_scale_shift;
    REPLACE_WITH_STR(model, "<biases offset=\"6\" size=\"6\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"18828\" size=\"128\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(YES);

    ASSERT_NO_THROW(ie.LoadNetwork(network, "kmb", config));
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsQuantizedConvolutionAfterScaleShift) {
    // TODO: Test fails. mcmCompiler can not compile the network (Convolution with bias). Jira: VPUNND-1474
    const std::string model = full_quant_model;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::map<std::string, std::string> config;
    details::CNNNetworkImplPtr clonedNetwork;

    setCommonConfig(config);

    // Parsing only is enabled because mcmCompiler can't compile layers.
    // TODO: turn off parsing only when mcmCompiler will be able to compile this layers.
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(YES);

    std::size_t weightSize = 147456 + 65536;
    std::size_t biasSize = 256 + 1024;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP32);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::FP32);

    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)network).getStats(&pstats, nullptr);

    ASSERT_EQ(StatusCode::OK, s);

    if (!pstats->isEmpty()) {
        clonedNetwork = cloneNet(network);
        details::CNNNetworkInt8Normalizer::NormalizeNetwork(*clonedNetwork, *pstats);

        ASSERT_NO_THROW(ie.LoadNetwork(CNNNetwork(clonedNetwork), "kmb", config));
    }
}

//  TODO: mcmCompiler assert: 'extendToK parameters dimensions doesn't match size of output_channels or 1'
//  JIRA Bug: VPUNND-1494
TEST_F(kmbLayersTests_nightly, DISABLED_TestsQuantizedConvolutionAfterScaleShiftNoBias) {
    std::string model = full_quant_model;

    REPLACE_WITH_STR(model, "<biases offset=\"147456\" size=\"256\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"213248\" size=\"1024\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::map<std::string, std::string> config;
    details::CNNNetworkImplPtr clonedNetwork;

    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(YES);

    std::size_t weightSize = 147456 + 65536;
    std::size_t biasSize = 256 + 1024;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP32);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::FP32);

    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)network).getStats(&pstats, nullptr);
    ASSERT_EQ(StatusCode::OK, s);

    if (!pstats->isEmpty()) {
        clonedNetwork = cloneNet(network);
        details::CNNNetworkInt8Normalizer::NormalizeNetwork(*clonedNetwork, *pstats);

        ASSERT_NO_THROW(ie.LoadNetwork(CNNNetwork(clonedNetwork), "kmb", config));
    }
}

std::vector<convolution_test_desc> convolution_only_fp16 = {
    {{1, 64, 64, 89}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 64, true, true, ""}, "FP16", "FP16", "FP16", 0,
        "FP16", "1x1_64x64x89to64x64x89", convolution_only_with_bias_template},
    {{1, 3, 224, 224}, {{2, 2}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, "", 1, 64, true, true, ""}, "FP16", "FP16", "FP16", 0,
        "FP16", "3x3_3x224x224to64x112x112", convolution_only_with_bias_template},
    {{1, 100, 100, 100}, {{1, 1}, {3, 3}, {1, 1}, {1, 1}, {1, 1}, "", 100, 100, true, true, ""}, "FP16", "FP16", "FP16",
        0, "FP16", "3x3_100x100x100_depthwise", convolution_only_with_bias_template},
    {{1, 3, 224, 224}, {{2, 2}, {7, 7}, {3, 3}, {3, 3}, {1, 1}, "", 1, 64, true, true, ""}, "FP16", "FP16", "FP16", 0,
        "FP16", "7x7_3x224x224to64x112x112", convolution_only_with_bias_template},
};

using ConvolutionFP16TestParam = testing::WithParamInterface<convolution_test_desc>;

class ConvolutionFP16Test : public ::testing::Test, public ConvolutionFP16TestParam {
public:
    using TestParam = ConvolutionFP16TestParam;

    static std::string getTestCaseName(TestParamInfo<ConvolutionFP16TestParam::ParamType> param) {
        auto testName = (param.param).test_name;
        std::replace(testName.begin(), testName.end(), '/', '_');
        std::replace(testName.begin(), testName.end(), '-', '_');
        return testName;
    }
};

TEST_P(ConvolutionFP16Test, fp16_convolution_only) {
    auto convTestParam = GetParam();

    Core ie;

    std::string blob_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    blob_name += ".blob";
    std::replace(blob_name.begin(), blob_name.end(), '/', '_');

    TBlob<uint8_t>::Ptr weightsBuffer = weightsBiasBlobPrepare(convTestParam);
#ifndef __arm__
    std::string model = instantiateConvTestIR(convTestParam);

    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setPrecision(Precision::FP16);

    auto outputsInfo = network.getOutputsInfo();
    outputsInfo["output"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);

    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, "kmb", config));
    ASSERT_NO_THROW(executableNetwork.Export(blob_name));
#else

#endif
}

INSTANTIATE_TEST_CASE_P(DISABLED_fp16_per_layer_compilation_fail, ConvolutionFP16Test,
    ::testing::ValuesIn(convolution_only_fp16), ConvolutionFP16Test::getTestCaseName);

#ifdef ENABLE_MCM_COMPILER

static void fillConvolutionIR(std::string& model, const convolution_test_params& params) {
    auto input_dims = params.input_dim;
    auto conv_params = params.conv_params;

    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsByteSize(input_dims, conv_params, "U8");
    size_t biasByteSize = output_dims[1] * sizeof(int32_t);

    REPLACE_WITH_NUM(model, "_CONV_INPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_CONV_INPUT_CHANNEL_", input_dims[1]);
    REPLACE_WITH_NUM(model, "_CONV_INPUT_HEIGHT_", input_dims[2]);
    REPLACE_WITH_NUM(model, "_CONV_INPUT_WIDTH_", input_dims[3]);

    REPLACE_WITH_NUM(model, "_CONV_WEIGHTS_BYTE_SIZE_", weightsByteSize);
    REPLACE_WITH_NUM(model, "_CONV_BIAS_BYTE_SIZE_", biasByteSize);
    REPLACE_WITH_NUM(model, "_CONV_BIAS_OFFSET_", weightsByteSize);

    // Assuming kernel is a square
    REPLACE_WITH_NUM(model, "_CONV_KERNEL_SIZE_", conv_params.kernel[0]);
    REPLACE_WITH_NUM_VECTOR(model, "_CONV_KERNEL_", conv_params.kernel);
    REPLACE_WITH_NUM_VECTOR(model, "_CONV_STRIDE_", conv_params.stride);

    REPLACE_WITH_NUM(model, "_CONV_OUTPUT_BATCH_", output_dims[0]);
    REPLACE_WITH_NUM(model, "_CONV_OUTPUT_CHANNEL_", output_dims[1]);
    REPLACE_WITH_NUM(model, "_CONV_OUTPUT_HEIGHT_", output_dims[2]);
    REPLACE_WITH_NUM(model, "_CONV_OUTPUT_WIDTH_", output_dims[3]);
}

class ConvolutionTest : public testing::WithParamInterface<convolution_test_params>, public kmbLayersTests_nightly {};

template <class Reference>
void InferAndCompare(ExecutableNetwork& exeNetwork, Reference refFunc, float tolerance) {
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first));
    auto data = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(data, inputBlob->byteSize(), static_cast<uint8_t>(1), static_cast<uint8_t>(1));

    ASSERT_NO_THROW(inferRequest.Infer());
    auto outputBlob = inferRequest.GetBlob(exeNetwork.GetOutputsInfo().begin()->first);
    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);

    auto refBlob = refFunc(inputBlob);
    Compare(refBlob, outputBlobFP32, tolerance);
}

TEST_P(ConvolutionTest, fq_convolution_only_manual) {
    // Besides weights and biases we need to store FQ blobs as well
    auto input_dims = GetParam().input_dim;
    auto conv_params = GetParam().conv_params;
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsByteSize(input_dims, conv_params, "FP32");
    size_t weightsSize = weightsByteSize / sizeof(float);
    size_t biasByteSize = conv_params.out_c * sizeof(float);
    size_t biasSize = biasByteSize / sizeof(float);

    // size_t quantizationParamsOffset = weightsByteSize + biasByteSize;
    size_t quantizationParamsByteSize = 48;
    size_t quantizationParamsSize = quantizationParamsByteSize / sizeof(float);

    auto weightsBuffer = make_shared_blob<uint8_t>(
        {Precision::U8, {quantizationParamsByteSize + weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<float*>();
    float* weightsDataStart = weightsBufferData + quantizationParamsSize;

    fillRealBuffer(weightsDataStart, weightsSize, static_cast<float>(1), static_cast<float>(1));
    for (size_t i = 0; i < weightsSize; ++i) {
        if (weightsDataStart[i] < 0.5f)
            weightsDataStart[i] = 0.0f;
        else
            weightsDataStart[i] = 1.0f;
    }

    std::fill_n(weightsDataStart + (weightsSize), biasSize, 0.0f);

    float* fqParamsData = weightsBufferData;

    // Parameters are hardcoded to force scale to be equal to 1
    float real_min_weight = 0.0f;
    float real_max_weight = 255.0f;

    // weights quantization params
    fqParamsData[0] = real_min_weight;
    fqParamsData[1] = real_max_weight;
    fqParamsData[2] = fqParamsData[0];
    fqParamsData[3] = fqParamsData[1];

    // Parameters are hardcoded to force scale to be equal to 1
    float min_input = 0.0f;
    float max_input = 255.0f;

    // output quantization params
    fqParamsData[4] = min_input;
    fqParamsData[5] = max_input;
    fqParamsData[6] = fqParamsData[4];
    fqParamsData[7] = fqParamsData[5];

    // input quantization params
    fqParamsData[8] = min_input;
    fqParamsData[9] = max_input;
    fqParamsData[10] = fqParamsData[8];
    fqParamsData[11] = fqParamsData[9];

    Core ie;

    convolution_test_desc allTestParams = {input_dims, conv_params, "FP32", "FP32", "FP32", quantizationParamsByteSize,
        "FP32", "", fq_convolution_only_slim};
    std::string model = instantiateConvTestIR(allTestParams);

    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);
    auto inputsInfo = network.getInputsInfo();
    inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["583"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));

    auto refFunc = [&](const Blob::Ptr& inputBlob) {
        auto inputBlobFP32 = ConvertU8ToFP32(inputBlob);
        auto outputBlob = exeNetwork.GetOutputsInfo()[exeNetwork.GetOutputsInfo().begin()->first];
        auto outputDesc = outputBlob->getTensorDesc();
        auto refOutputBlob = make_shared_blob<float>({Precision::FP32, output_dims, outputDesc.getLayout()});
        refOutputBlob->allocate();
        auto m_data = refOutputBlob->buffer().as<uint8_t*>();
        std::fill(m_data, m_data + refOutputBlob->byteSize(), 0);

        auto weightsData = weightsDataStart;
        auto bias_data = conv_params.with_bias ? (weightsData + weightsSize) : nullptr;
        ref_conv_common({inputBlobFP32}, *refOutputBlob, weightsData, weightsSize, bias_data, biasSize, conv_params);
        testOverflow<float, uint8_t>(refOutputBlob);

        return refOutputBlob;
    };

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first));
    IE_ASSERT(inputBlob->getTensorDesc().getPrecision() == Precision::U8);

    auto data = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(data, inputBlob->byteSize(), static_cast<uint8_t>(1), static_cast<uint8_t>(1));

    auto refBlob = refFunc(inputBlob);

    ASSERT_NO_THROW(inferRequest.Infer());
    auto outputBlob = inferRequest.GetBlob(exeNetwork.GetOutputsInfo().begin()->first);
    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);

    {                           // Output need to be dequantized for comparing with reference
        float levels = 255.0f;  // Need to be synchronized with kmb plugin
        float outputScale = (max_input - min_input) / levels;
        float zeroPoint = 0.0f;  // Need to be synchronized with kmb plugin

        auto outputData = outputBlobFP32->buffer().as<float*>();
        for (size_t i = 0; i < outputBlobFP32->size(); ++i) {
            outputData[i] = outputScale * (outputData[i] - zeroPoint);
        }
    }

    Compare(outputBlobFP32, refBlob, 0.1f);
}

TEST_P(ConvolutionTest, u8_convolution_only_manual) {
    auto input_dims = GetParam().input_dim;
    auto conv_params = GetParam().conv_params;
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsSize(input_dims, conv_params, "U8");
    size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    size_t biasSize = biasByteSize / sizeof(int32_t);

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    std::fill_n(weightsBufferData, weightsSize, static_cast<uint8_t>(1));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    std::fill_n(biasData, biasSize, static_cast<uint32_t>(1));

    convolution_test_desc allTestParams = {
        input_dims, conv_params, "U8", "U8", "U8", 0, "I32", "", convolution_only_with_bias_template};
    std::string model = instantiateConvTestIR(allTestParams);

    Core ie;
    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["output"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());
    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first));
    auto data = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(data, inputBlob->byteSize(), static_cast<uint8_t>(1), static_cast<uint8_t>(1));

    auto weightsData = weightsBuffer->buffer().as<int8_t*>();
    auto bias_data = conv_params.with_bias ? reinterpret_cast<int32_t*>(weightsData + weightsSize) : nullptr;

    auto outputBlob = inferRequest.GetBlob(exeNetwork.GetOutputsInfo().begin()->first);
    auto outputDesc = outputBlob->getTensorDesc();
    auto refOutputBlob = make_shared_blob<float>({Precision::FP32, outputDesc.getDims(), outputDesc.getLayout()});
    refOutputBlob->allocate();
    data = refOutputBlob->buffer().as<uint8_t*>();
    std::fill(data, data + refOutputBlob->byteSize(), 0);

    ref_conv_common({inputBlob}, *refOutputBlob, weightsData, weightsSize, bias_data, biasSize, conv_params);
    testOverflow<float, uint8_t>(refOutputBlob);
    ASSERT_NO_THROW(inferRequest.Infer());
    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);

    Compare(refOutputBlob, outputBlobFP32, 1.1f);
}

TEST_P(ConvolutionTest, convolution_and_relu_u8) {
    auto input_dims = GetParam().input_dim;
    auto conv_params = GetParam().conv_params;
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsSize(input_dims, conv_params, "U8");
    size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    size_t biasSize = biasByteSize / sizeof(int32_t);

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    fillIntBuffer(weightsBufferData, weightsSize, static_cast<uint8_t>(1), static_cast<uint8_t>(1));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    fillIntBuffer(biasData, biasSize, static_cast<uint32_t>(1), static_cast<uint32_t>(1));

    std::string model = conv_relu_u8_test;
    fillConvolutionIR(model, {input_dims, conv_params});

    Core ie;
    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["relu"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    InferenceEngine::ExecutableNetwork exeNetwork;
    (exeNetwork = ie.LoadNetwork(network, "KMB", config));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first));
    auto inputDesc = inputBlob->getTensorDesc();
    auto data = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(data, inputBlob->byteSize(), static_cast<uint8_t>(1), static_cast<uint8_t>(1));

    int8_t* weightsData = weightsBuffer->buffer().as<int8_t*>();
    int32_t* bias_data = conv_params.with_bias ? reinterpret_cast<int32_t*>(weightsData + weightsSize) : nullptr;

    auto outputBlob = inferRequest.GetBlob(exeNetwork.GetOutputsInfo().begin()->first);
    auto outputDesc = outputBlob->getTensorDesc();
    auto refOutputBlob = make_shared_blob<float>({Precision::FP32, outputDesc.getDims(), outputDesc.getLayout()});
    refOutputBlob->allocate();
    data = refOutputBlob->buffer().as<uint8_t*>();
    std::fill(data, data + refOutputBlob->byteSize(), 0);

    ref_conv_common({inputBlob}, *refOutputBlob, weightsData, weightsSize, bias_data, biasSize, conv_params);
    ref_ReLU<uint8_t>(refOutputBlob);
    testOverflow<float, uint8_t>(refOutputBlob);

    ASSERT_NO_THROW(inferRequest.Infer());

    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);
    Compare(refOutputBlob, outputBlobFP32, 1.1f);
}

// Assuming input and output are in NCHW layout
// All parameters after kernel must be consistent with IR
// {input_dim}, {stride}, {kernel}, {pads_begin}, {pads_end}, {dilation}, "", group, out_c, with_bias, with_weights,
// quantization_level};
std::vector<convolution_test_params> test_params = {
    {{1, 3, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}},
    {{1, 8, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 64, true, true, ""}},
    // {{1, 64, 16, 16}, {{1, 1}, {2, 2}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false, true, ""}},
};

const std::vector<InferenceEngine::Layout> test_layouts = {
    InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCHW};

INSTANTIATE_TEST_CASE_P(accuracy, ConvolutionTest, ::testing::ValuesIn(test_params));

// These parameters show that compilation crashes when number of output channels is not factor of 16
std::vector<convolution_test_params> sigsegv_due_to_low_output_channels_number_test_params = {
    {{1, 3, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 3, true, true, ""}},
    {{1, 3, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 9, true, true, ""}},
    {{1, 3, 16, 16}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 20, true, true, ""}},
};

INSTANTIATE_TEST_CASE_P(DISABLED_accuracy_low_OC, ConvolutionTest,
    ::testing::ValuesIn(sigsegv_due_to_low_output_channels_number_test_params));

struct convolution_input_fill_test_params {
    SizeVector input_dim;
    CommonTestUtils::conv_common_params conv_params;
    void (*fillBlob)(Blob::Ptr&, uint8_t);
    uint8_t fillSeed;
};

void fillBlobWithSameValue(Blob::Ptr& inputBlob, uint8_t val) {
    uint8_t* inDataU8 = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(inDataU8, inputBlob->byteSize(), static_cast<uint8_t>(val), static_cast<uint8_t>(val));
}

// Each channel will have seed value multiplied by the number of channel
void fillBlobChannelByChannel(Blob::Ptr& inputBlob, uint8_t val) {
    uint8_t* inDataU8 = inputBlob->buffer().as<uint8_t*>();
    const auto input_dims = inputBlob->getTensorDesc().getDims();
    const size_t planeSize = input_dims[2] * input_dims[3];
    const size_t channelSize = input_dims[1] * planeSize;
    for (size_t n = 0; n < input_dims[0]; ++n) {
        for (size_t c = 0; c < input_dims[1]; ++c) {
            IE_ASSERT(n * channelSize + c * planeSize < inputBlob->byteSize());
            fillIntBuffer(inDataU8 + n * channelSize + c * planeSize, planeSize, static_cast<uint8_t>(val * c),
                static_cast<uint8_t>(val * c));
        }
    }
}

class ConvolutionTestIdent :
    public testing::WithParamInterface<convolution_input_fill_test_params>,
    public kmbLayersTests_nightly {};

TEST_P(ConvolutionTestIdent, u8_convolution_identity) {
    auto input_dims = GetParam().input_dim;
    auto conv_params = GetParam().conv_params;
    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    size_t weightsByteSize = getConvWeightsSize(input_dims, conv_params, "U8");
    size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    size_t biasSize = biasByteSize / sizeof(int32_t);

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    uint8_t* weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    std::fill_n(weightsBufferData, weightsSize, static_cast<uint8_t>(0));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    std::fill_n(biasData, biasSize, static_cast<uint32_t>(0));

    convolution_test_desc allTestParams = {
        input_dims, conv_params, "U8", "U8", "U8", 0, "I32", "", convolution_only_with_bias_template};
    std::string model = instantiateConvTestIR(allTestParams);

    Core ie;
    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);
    _inputsInfo["input"]->setLayout(Layout::NCHW);

    CNNLayerPtr convPtr = network.getLayerByName("output");
    ConvolutionLayer* convLayer = dynamic_cast<ConvolutionLayer*>(convPtr.get());

    // build identity kernel which transforms the tensor to copy of itself
    DataPtr kernelWeights = convLayer->insData[1].lock();
    CNNLayer::Ptr kernelWeightsLayer = kernelWeights->getCreatorLayer().lock();
    Blob::Ptr kernelWeightsBlob = kernelWeightsLayer->blobs.begin()->second;
    IE_ASSERT(kernelWeightsBlob != nullptr);
    IE_ASSERT(kernelWeightsBlob->getTensorDesc().getLayout() == Layout::NCHW);
    fillDiagonalKernel(kernelWeightsBlob, static_cast<uint8_t>(1u));

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["output"]->setPrecision(Precision::U8);
    _outputsInfo["output"]->setLayout(Layout::NCHW);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    InferenceEngine::ExecutableNetwork exeNetwork;
    exeNetwork = ie.LoadNetwork(network, "KMB", config);
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());
    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first));

    // fill input with some meaningful data
    GetParam().fillBlob(inputBlob, static_cast<uint8_t>(GetParam().fillSeed));

    int8_t* weightsData = weightsBuffer->buffer().as<int8_t*>();
    int32_t* bias_data = conv_params.with_bias ? reinterpret_cast<int32_t*>(weightsData + weightsSize) : nullptr;

    auto outputBlobU8 = inferRequest.GetBlob(exeNetwork.GetOutputsInfo().begin()->first);
    auto outputDesc = outputBlobU8->getTensorDesc();

    {
        // calc reference FP32 output to make sure it will not overflow U8 range
        auto refOutputBlobFP32 =
            make_shared_blob<float>({Precision::FP32, outputDesc.getDims(), outputDesc.getLayout()});
        refOutputBlobFP32->allocate();
        uint8_t* refOutDataU8 = refOutputBlobFP32->buffer().as<uint8_t*>();
        std::fill(refOutDataU8, refOutDataU8 + refOutputBlobFP32->byteSize(), 0);

        ref_conv_common({inputBlob}, *refOutputBlobFP32, weightsData, weightsSize, bias_data, biasSize, conv_params);

        testOverflow<float, uint8_t>(refOutputBlobFP32);
    }

    // calc reference U8 output
    auto refOutputBlobU8 = make_shared_blob<uint8_t>({Precision::U8, outputDesc.getDims(), outputDesc.getLayout()});
    refOutputBlobU8->allocate();
    uint8_t* refOutDataU8 = refOutputBlobU8->buffer().as<uint8_t*>();
    std::fill(refOutDataU8, refOutDataU8 + refOutputBlobU8->byteSize(), 0);

    ref_conv_common_prec(
        {inputBlob}, *refOutputBlobU8, weightsData, weightsSize, bias_data, biasSize, conv_params, Precision::U8);

    // make sure that tensor identity is preserved for reference output
    CompareCommonExactTyped<uint8_t>(inputBlob, refOutputBlobU8);

    const Blob::Ptr refOutputBlobFP32 = ConvertU8ToFP32(refOutputBlobU8);
    testOverflow<float, uint8_t>(refOutputBlobFP32);

    ASSERT_NO_THROW(inferRequest.Infer());

    // make sure that tensor identity is preserved for device output
    if (!hasFakeXLinkDevice()) {
        CompareCommonExactTyped<uint8_t>(inputBlob, outputBlobU8);
    }
}

// Assuming input and output are in NCHW layout
// All parameters after kernel must be consistent with IR
// {input_dim}, {stride}, {kernel}, {pads_begin}, {pads_end}, {dilation}, "", group, out_c, with_bias, with_weights,
// quantization_level};
std::vector<convolution_input_fill_test_params> test_params_identity_kernel_1x1 = {
    {{1, 16, 16, 16}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobWithSameValue, 1},
    {{1, 16, 32, 32}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobWithSameValue, 1},
    {{1, 16, 20, 20}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobWithSameValue, 1},
    {{1, 16, 18, 20}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobWithSameValue, 1},
    {{1, 16, 18, 20}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobWithSameValue, 2},
    {{1, 16, 16, 16}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobChannelByChannel, 1},
    {{1, 16, 32, 32}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobChannelByChannel, 1},
    {{1, 16, 20, 20}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobChannelByChannel, 1},
    {{1, 16, 18, 20}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobChannelByChannel, 1},
    {{1, 16, 18, 20}, {{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, "", 1, 16, true, true, ""}, fillBlobChannelByChannel, 2},
};

INSTANTIATE_TEST_CASE_P(accuracy, ConvolutionTestIdent, ::testing::ValuesIn(test_params_identity_kernel_1x1));

struct convolution_and_pooling_test_params {
    SizeVector input_dim;
    CommonTestUtils::conv_common_params conv_params;
    CommonTestUtils::pool_common_params pool_params;
    bool is_positive_weights;
};

static void fillConvAndPoolIR(std::string& model, const convolution_and_pooling_test_params& params) {
    auto input_dims = params.input_dim;
    auto conv_params = params.conv_params;
    auto pool_params = params.pool_params;

    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    SizeVector pool_input_dims = output_dims;
    SizeVector pool_output_dims;
    CommonTestUtils::getPoolOutShape(pool_input_dims, pool_params, pool_output_dims);

    fillConvolutionIR(model, {input_dims, conv_params});

    REPLACE_WITH_NUM(model, "_POOL_INPUT_BATCH_", pool_input_dims[0]);
    REPLACE_WITH_NUM(model, "_POOL_INPUT_CHANNEL_", pool_input_dims[1]);
    REPLACE_WITH_NUM(model, "_POOL_INPUT_HEIGHT_", pool_input_dims[2]);
    REPLACE_WITH_NUM(model, "_POOL_INPUT_WIDTH_", pool_input_dims[3]);

    REPLACE_WITH_NUM(model, "_POOL_OUTPUT_BATCH_", pool_output_dims[0]);
    REPLACE_WITH_NUM(model, "_POOL_OUTPUT_CHANNEL_", pool_output_dims[1]);
    REPLACE_WITH_NUM(model, "_POOL_OUTPUT_HEIGHT_", pool_output_dims[2]);
    REPLACE_WITH_NUM(model, "_POOL_OUTPUT_WIDTH_", pool_output_dims[3]);

    REPLACE_WITH_NUM_VECTOR(model, "_POOL_KERNEL_", pool_params.kernel);
    REPLACE_WITH_NUM_VECTOR(model, "_POOL_STRIDE_", pool_params.stride);
    REPLACE_WITH_STR(model, "_POOL_EXCLUDE_PAD_", pool_params.exclude_pad ? "true" : "false");
}

class ConvolutionAndPoolingTest :
    public testing::WithParamInterface<convolution_and_pooling_test_params>,
    public kmbLayersTests_nightly {};

TEST_P(ConvolutionAndPoolingTest, DISABLED_convolution_and_pooling_u8) {
    if (!GetParam().is_positive_weights) SKIP();  // TODO
    std::string model = conv_pool_u8_test;

    auto input_dims = GetParam().input_dim;
    auto conv_params = GetParam().conv_params;
    auto pool_params = GetParam().pool_params;
    std::string weight_precision_str = "U8";

    SizeVector output_dims;
    CommonTestUtils::getConvOutShape(input_dims, conv_params, output_dims);

    SizeVector pool_input_dims = output_dims;
    SizeVector pool_output_dims;
    CommonTestUtils::getPoolOutShape(pool_input_dims, pool_params, pool_output_dims);

    size_t weightsByteSize = getConvWeightsSize(input_dims, conv_params, weight_precision_str);
    size_t weightsSize = weightsByteSize / sizeof(uint8_t);

    size_t biasByteSize = output_dims[1] * sizeof(int32_t);
    size_t biasSize = biasByteSize / sizeof(int32_t);

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<int8_t*>();
    int8_t weight_val = 1;
    if (!GetParam().is_positive_weights) weight_val = -1;

    fillIntBuffer(weightsBufferData, weightsSize, weight_val, weight_val);

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    fillIntBuffer(biasData, biasSize, static_cast<uint32_t>(1), static_cast<uint32_t>(1));

    REPLACE_WITH_STR(model, "_WEIGHT_PRECISION_", weight_precision_str);

    fillConvAndPoolIR(model, {input_dims, conv_params, pool_params});

    Core ie;
    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["pooling"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first));
    auto inputDesc = inputBlob->getTensorDesc();
    auto data = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(data, inputBlob->byteSize(), static_cast<uint8_t>(1), static_cast<uint8_t>(1));

    int8_t* weightsData = weightsBuffer->buffer().as<int8_t*>();
    int32_t* bias_data = conv_params.with_bias ? reinterpret_cast<int32_t*>(weightsData + weightsSize) : nullptr;

    auto outputBlob = inferRequest.GetBlob(exeNetwork.GetOutputsInfo().begin()->first);
    auto outputDesc = outputBlob->getTensorDesc();

    Blob::Ptr refConvOutputBlob = make_shared_blob<float>({Precision::FP32, output_dims, outputDesc.getLayout()});
    refConvOutputBlob->allocate();
    data = refConvOutputBlob->buffer().as<uint8_t*>();
    std::fill(data, data + refConvOutputBlob->byteSize(), 0);
    ref_conv_common({inputBlob}, *refConvOutputBlob, weightsData, weightsSize, bias_data, biasSize, conv_params);

    Blob::Ptr refOutputBlob = make_shared_blob<float>({Precision::FP32, outputDesc.getDims(), outputDesc.getLayout()});
    refOutputBlob->allocate();
    data = refOutputBlob->buffer().as<uint8_t*>();
    std::fill(data, data + refOutputBlob->byteSize(), 0);
    ref_pool_common<float>({refConvOutputBlob}, *refOutputBlob, pool_params);

    ASSERT_NO_THROW(inferRequest.Infer());

    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);
    Compare(refOutputBlob, outputBlobFP32, 1.1f);
}

// Assuming input and output are in NCHW layout
// All parameters after kernel must be consistent with IR
// {input_dim}, {stride}, {kernel}, {pads_begin}, {pads_end}, {dilation}, "", group, out_c, with_bias, with_weights,
// quantization_level};
std::vector<convolution_and_pooling_test_params> conv_and_pool_params = {
    {{1, 1, 64, 64}, {{1, 1}, {2, 2}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false, true, ""},
        {{2, 2}, {2, 2}, {0, 0}, {0, 0}, "same_upper", false, "true"}, true},
    {{1, 1, 128, 128}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 64, false, true, ""},
        {{2, 2}, {4, 4}, {0, 0}, {0, 0}, "same_upper", false, "true"}, true},
    {{1, 1, 64, 64}, {{1, 1}, {2, 2}, {0, 0}, {0, 0}, {1, 1}, "", 1, 256, false, true, ""},
        {{2, 2}, {2, 2}, {0, 0}, {0, 0}, "same_upper", false, "true"}, false},
    {{1, 1, 128, 128}, {{1, 1}, {3, 3}, {0, 0}, {0, 0}, {1, 1}, "", 1, 64, false, true, ""},
        {{2, 2}, {4, 4}, {0, 0}, {0, 0}, "same_upper", false, "true"}, false},
};

INSTANTIATE_TEST_CASE_P(accuracy, ConvolutionAndPoolingTest, ::testing::ValuesIn(conv_and_pool_params));

#endif
