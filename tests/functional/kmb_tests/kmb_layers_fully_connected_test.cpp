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

#include <blob_factory.hpp>
#include <limits>

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"
#include "quantization_helpers.hpp"
#include "test_model/kmb_test_fake_quantize_def.hpp"

// TODO https://jira.devtools.intel.com/browse/VPUNND-2304
//  resnet-50 has 2d output, but mcmCompiler returns 4d one, therefore tests failed with assert (inDims == outDims)

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

typedef std::tuple<tensor_test_params, tensor_test_params, size_t> fully_connected_test_params;
typedef kmbLayerTestBaseWithParam<fully_connected_test_params> kmbLayersTestsFullyConnectedParams;

#ifdef ENABLE_MCM_COMPILER
TEST_P(kmbLayersTestsFullyConnectedParams, DISABLED_TestsFullyConnected) {
    auto param = GetParam();
    tensor_test_params inputTensor = std::get<0>(param);
    tensor_test_params outputTensor = std::get<1>(param);
    size_t outSize = std::get<2>(param);

    size_t weightsSize = outSize * inputTensor.n * inputTensor.c * inputTensor.h * inputTensor.w * sizeof(uint16_t);
    size_t biasesSize = 0;

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    std::cout << ::testing::UnitTest::GetInstance()->current_test_info()->name()
              << " test_info->name()=" << test_info->name() << " test_info->test_case_name() "
              << test_info->test_case_name() << std::endl;

    std::map<std::string, std::string> params;
    params["out-size"] = std::to_string(outSize);

    Blob::Ptr weightsBlob(GenWeights<uint16_t>(weightsSize + biasesSize));

    SetInputTensor(inputTensor);
    SetOutputTensor(outputTensor);
    NetworkInit(
        "FullyConnected", &params, weightsSize, biasesSize, std::static_pointer_cast<TBlob<uint8_t>>(weightsBlob),
        Precision::FP16  // output precision
    );
}

static const fully_connected_test_params paramsTable[] = {
    std::make_tuple<tensor_test_params, tensor_test_params, size_t>({1, 128, 2, 2},  // input tensor
        {1, 1024, 1, 1},                                                             // output tensor
        128                                                                          // out-size
        ),
    std::make_tuple<tensor_test_params, tensor_test_params, size_t>({1, 64, 2, 2},  // input tensor
        {1, 2048, 1, 1},                                                            // output tensor
        64                                                                          // out-size
        ),
};

INSTANTIATE_TEST_CASE_P(loadNetworkNoThrow, kmbLayersTestsFullyConnectedParams, ::testing::ValuesIn(paramsTable));
struct fullyConnected_test_params {
    SizeVector input_dim;
    uint32_t out_channels;
    bool with_bias;
    bool with_weights;
    std::string quantization_level;
};

size_t getFCWeightsByteSize(const size_t inChannels, const size_t outChannels, const std::string& precision) {
    size_t type_size = 1lu;
    if (precision == "FP32")
        type_size = sizeof(float);
    else if (precision == "FP16") {
        type_size = sizeof(ie_fp16);
    }
    return inChannels * outChannels * type_size;
}

template <typename wei_data_t, typename bias_data_t>
void ref_fc(const Blob::Ptr src, const wei_data_t* weights, const size_t weightsSize, const bias_data_t* biases,
    const size_t biasSize, Blob::Ptr dst, uint32_t out_c, SizeVector input_dims) {
    size_t IW = 1;
    size_t IH = 1;
    size_t IC = 1;

    size_t OC = out_c;

    switch (src->getTensorDesc().getLayout()) {
    case NCHW:
        IW = input_dims[3];
        IH = input_dims[2];
        IC = input_dims[1];
        break;
    case NHWC:
        IC = input_dims[3];
        IW = input_dims[2];
        IH = input_dims[1];
        break;
    case NC:
        IC = input_dims[1];
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported layout: " << src->getTensorDesc().getLayout();
    }

    auto* dst_data = dst->buffer().as<float*>();
    Precision src_precision = src->getTensorDesc().getPrecision();

    IE_ASSERT(IW * IH * IC * OC == weightsSize);
    IE_ASSERT(OC == dst->getTensorDesc().getDims()[1]);

    for (size_t oc = 0; oc < OC; oc++) {
        if (biases != nullptr) dst_data[oc] = biases[oc];
        for (size_t ic = 0; ic < IC; ic++) {
            for (size_t kh = 0; kh < IH; kh++) {
                for (size_t kw = 0; kw < IW; kw++) {
                    size_t iidx = ic * IH * IW + kh * IW + kw;
                    size_t widx = oc * IC * IH * IW + ic * IH * IW + kh * IW + kw;

                    if (src_precision == Precision::U8) {
                        dst_data[oc] += (src->cbuffer().as<const uint8_t*>())[iidx] * weights[widx];
                    } else if (src_precision == Precision::I8) {
                        dst_data[oc] += (src->cbuffer().as<const int8_t*>())[iidx] * weights[widx];
                    } else if (src_precision == Precision::FP32) {
                        dst_data[oc] += (src->cbuffer().as<const float*>())[iidx] * weights[widx];
                    } else {
                        THROW_IE_EXCEPTION << "Unsupported precision: " << src_precision;
                    }
                }
            }
        }
    }
}

static void fillFcIR(std::string& model, SizeVector input_dims, size_t weightsBufferOffset, size_t weightsByteSize,
    size_t biasBufferOffset, size_t biasByteSize, uint32_t out_channels, bool withFQ = false) {
    REPLACE_WITH_NUM(model, "_INPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_INPUT_CHANNEL_", input_dims[1]);

    if (input_dims.size() == 4 && !withFQ) {
        REPLACE_WITH_NUM(model, "_INPUT_HEIGHT_", input_dims[2]);
        REPLACE_WITH_NUM(model, "_INPUT_WIDTH_", input_dims[3]);
    }

    REPLACE_WITH_NUM(model, "_WEIGHTS_OFFSET_", weightsBufferOffset);
    REPLACE_WITH_NUM(model, "_WEIGHTS_BYTE_SIZE_", weightsByteSize);

    REPLACE_WITH_NUM(model, "_BIAS_OFFSET_", biasBufferOffset);
    REPLACE_WITH_NUM(model, "_BIAS_BYTE_SIZE_", biasByteSize);

    REPLACE_WITH_NUM(model, "_OUTPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_OUTPUT_CHANNEL_", out_channels);
}

typedef kmbLayerTestBaseWithParam<fullyConnected_test_params> kmbLayersTestsFullyConnectedWithIR;

Blob::Ptr getQuantizedBlob(Blob::Ptr blob, size_t blobSize, float min, float max)
{
    double inf = std::numeric_limits<double>::infinity();
    double scales = (max - min) / 255;
    int64_t zeroPoint =
        vpu::KmbPlugin::KmbQuantizationHelpers::calculateZeroPoint(max, min, 256, Precision::U8);
    return vpu::KmbPlugin::KmbQuantizationHelpers::quantizeBlob(
        blob, {1, blobSize, 1, 1}, {{zeroPoint}, {scales}, {-inf}, {inf}}, Precision::U8, " ");
}

TEST_P(kmbLayersTestsFullyConnectedWithIR, fc_only) {
    size_t weightsBufferOffset = 48;
    auto input_dims = GetParam().input_dim;
    uint32_t outChannels = GetParam().out_channels;

    size_t weightsByteSize = getFCWeightsByteSize(input_dims[1], outChannels, "FP32");
    size_t weightsSize = weightsByteSize / sizeof(float);
    size_t biasByteSize = outChannels * sizeof(float);
    size_t biasSize = outChannels;

    float min_weight = 0.0f;
    float max_weight = 1.0f;

    auto weightsBuffer =
        make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize + weightsBufferOffset}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<float*>();
    fillRealBuffer(weightsBufferData, weightsBuffer->byteSize() / sizeof(float), min_weight, max_weight);
    float* weightsDataStart = weightsBufferData + weightsBufferOffset / sizeof(float);

    auto weightsData = weightsDataStart;
    auto bias_data = GetParam().with_bias ? weightsData + weightsSize : nullptr;

    auto weights_blob = getQuantizedBlob(weightsBuffer, weightsSize, min_weight, max_weight);

    float min_input = 0.0f;
    float max_input = 1.0f;

    const auto desc = TensorDesc(Precision::U8, input_dims, TensorDesc::getLayoutByDims(input_dims));
    auto inBlobForReference = make_blob_with_precision(desc);
    inBlobForReference->allocate();
    auto inputData = inBlobForReference->buffer().as<uint8_t*>();
    fillIntBuffer(
        inputData, inBlobForReference->byteSize(), static_cast<uint8_t>(min_input), static_cast<uint8_t>(max_input));

    auto inputBlobFP32 = ConvertU8ToFP32(inBlobForReference);

    auto refOutputBlob = make_shared_blob<float>({Precision::FP32, {1, outChannels}, Layout::NC});
    refOutputBlob->allocate();
    auto data_ref = refOutputBlob->buffer().as<uint8_t*>();
    std::fill(data_ref, data_ref + refOutputBlob->byteSize(), 0);

    // calc reference blob
    ref_fc(inputBlobFP32, weightsData, weightsSize, bias_data, biasSize, refOutputBlob, outChannels, input_dims);

    auto result = std::minmax_element(
        refOutputBlob->buffer().as<float*>(), refOutputBlob->buffer().as<float*>() + refOutputBlob->size());

    float min_output = *result.first;
    float max_output = *result.second;

    auto fq_ref_blob = getQuantizedBlob(refOutputBlob, outChannels, min_output, max_output);

    float* fqParamsData = weightsBufferData;

    // weights quantization params
    fqParamsData[0] = min_weight;
    fqParamsData[1] = max_weight;
    fqParamsData[2] = fqParamsData[0];
    fqParamsData[3] = fqParamsData[1];

    // output quantization params
    fqParamsData[4] = min_output;
    fqParamsData[5] = max_output;
    fqParamsData[6] = fqParamsData[4];
    fqParamsData[7] = fqParamsData[5];

    // input quantization params
    fqParamsData[8] = min_input;
    fqParamsData[9] = max_input;
    fqParamsData[10] = fqParamsData[8];
    fqParamsData[11] = fqParamsData[9];

    std::string model = fq_fully_connected_only_slim;

    fillFcIR(model, input_dims, weightsBufferOffset,
        weightsByteSize, weightsBufferOffset + weightsByteSize, biasByteSize, outChannels, true);

    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);
    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2/after_quant/FakeQuantWithMinMaxVars"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);

    Core ie;
    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, "kmb", config));
    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(executableNetwork.GetInputsInfo().begin()->first));
    auto data = inputBlob->buffer().as<uint8_t*>();
    for (size_t i = 0; i < inputBlob->size(); i++) {
        data[i] = inputData[i];
    }

    auto outputBlob = inferRequest.GetBlob(executableNetwork.GetOutputsInfo().begin()->first);
    ASSERT_NO_THROW(inferRequest.Infer());

    Blob::Ptr deqOutput = dequantize(outputBlob, min_output, max_output, 256);
    Blob::Ptr deqRef = dequantize(fq_ref_blob, min_output, max_output, 256);

    Compare(deqRef, deqOutput, 1.1f);
}

TEST_P(kmbLayersTestsFullyConnectedWithIR, fc_only_u8) {
    auto input_dims = GetParam().input_dim;
    uint32_t outChannels = GetParam().out_channels;

    size_t weightsByteSize = getFCWeightsByteSize(input_dims[1], outChannels, "U8");
    size_t weightsSize = weightsByteSize / sizeof(uint8_t);
    size_t biasByteSize = outChannels * sizeof(uint32_t);
    size_t biasSize = outChannels;

    auto weightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<uint8_t*>();
    fillIntBuffer(weightsBufferData, weightsByteSize, static_cast<uint8_t>(0), static_cast<uint8_t>(4));

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsSize);
    std::fill_n(biasData, biasSize, static_cast<uint32_t>(1));

    Core ie;

    std::string blob_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    blob_name += ".blob";
    std::replace(blob_name.begin(), blob_name.end(), '/', '_');

    std::string model;

    if (input_dims.size() == 4) {
        model = fc_u8_only_4d;
    } else {
        model = fc_u8_only;
    }

    fillFcIR(model, input_dims, 0, weightsByteSize, weightsByteSize, biasByteSize, outChannels);

    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);

    ExecutableNetwork executableNetwork;
    (executableNetwork = ie.LoadNetwork(network, "KMB", config));
    ASSERT_NO_THROW(executableNetwork.Export(blob_name));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = executableNetwork.CreateInferRequest());

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(executableNetwork.GetInputsInfo().begin()->first));
    auto data = inputBlob->buffer().as<uint8_t*>();
    fillIntBuffer(data, inputBlob->byteSize(), static_cast<uint8_t>(0), static_cast<uint8_t>(4));

    auto weightsData = weightsBuffer->buffer().as<uint8_t*>();
    auto bias_data = GetParam().with_bias ? reinterpret_cast<int32_t*>(weightsData + weightsSize) : nullptr;

    auto outputBlob = inferRequest.GetBlob(executableNetwork.GetOutputsInfo().begin()->first);
    auto outputDesc = outputBlob->getTensorDesc();

    auto refOutputBlob = make_shared_blob<float>({Precision::FP32, outputDesc.getDims(), outputDesc.getLayout()});
    refOutputBlob->allocate();
    data = refOutputBlob->buffer().as<uint8_t*>();
    std::fill(data, data + refOutputBlob->byteSize(), 0);

    ref_fc(inputBlob, weightsData, weightsSize, bias_data, biasSize, refOutputBlob, outChannels, input_dims);

    ASSERT_NO_THROW(inferRequest.Infer());
    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);

    Compare(refOutputBlob, outputBlobFP32, 1.1f);
}
// Assuming input and output are in NCHW layout
// {input_dim}, out_c, with_bias, with_weights, quantization_level};
std::vector<fullyConnected_test_params> test_params_fc = {
    {{1, 16, 1, 1}, 16, false, true, ""},
    {{1, 8, 1, 1}, 16, false, true, ""},
    {{1, 16}, 8, false, true, ""},
};

INSTANTIATE_TEST_CASE_P(accuracy, kmbLayersTestsFullyConnectedWithIR, ::testing::ValuesIn(test_params_fc));
#endif
