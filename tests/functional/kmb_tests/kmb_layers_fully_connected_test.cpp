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

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"

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

void ref_fc(const Blob::Ptr src, const uint8_t* weights, const size_t weightsSize, const int32_t* biases,
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

    const auto* src_data = src->buffer().as<const uint8_t*>();
    auto* dst_data = dst->buffer().as<float*>();

    IE_ASSERT(IW * IH * IC * OC == weightsSize);
    IE_ASSERT(OC == dst->getTensorDesc().getDims()[1]);

    for (size_t oc = 0; oc < OC; oc++) {
        if (biases != nullptr) dst_data[oc] = biases[oc];
        for (size_t ic = 0; ic < IC; ic++) {
            for (size_t kh = 0; kh < IH; kh++) {
                for (size_t kw = 0; kw < IW; kw++) {
                    size_t iidx = ic * IH * IW + kh * IW + kw;
                    size_t widx = oc * IC * IH * IW + ic * IH * IW + kh * IW + kw;

                    dst_data[oc] += src_data[iidx] * weights[widx];
                }
            }
        }
    }
}

static void fillFcIR(std::string& model, SizeVector input_dims, size_t weightsBufferOffset, size_t weightsByteSize,
    size_t biasBufferOffset, size_t biasByteSize, uint32_t out_channels) {
    REPLACE_WITH_NUM(model, "_INPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_INPUT_CHANNEL_", input_dims[1]);

    if (input_dims.size() == 4) {
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

TEST_P(kmbLayersTestsFullyConnectedWithIR, DISABLED_fc_only) {
    // Besides weights and biases we need to store FQ blobs as well
    size_t weightsBufferOffset = 48;
    auto input_dims = GetParam().input_dim;
    uint32_t outChannels = GetParam().out_channels;

    size_t weightsByteSize = getFCWeightsByteSize(input_dims[1], outChannels, "FP32");
    size_t weightsSize = weightsByteSize / sizeof(float);
    size_t biasByteSize = outChannels * sizeof(float);
    size_t biasSize = outChannels;

    auto weightsBuffer =
        make_shared_blob<uint8_t>({Precision::U8, {weightsByteSize + biasByteSize + weightsBufferOffset}, Layout::C});
    weightsBuffer->allocate();
    auto weightsBufferData = weightsBuffer->buffer().as<float*>();
    std::fill(
        weightsBufferData, weightsBufferData + (weightsSize + biasSize + weightsBufferOffset / sizeof(float)), 1.0f);

    Core ie;

    std::string blob_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    blob_name += ".blob";
    std::replace(blob_name.begin(), blob_name.end(), '/', '_');

    std::string model = fq_fully_connected_only_slim;

    REPLACE_WITH_NUM(model, "_INPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_INPUT_CHANNEL_", input_dims[1]);

    REPLACE_WITH_NUM(model, "_WEIGHTS_OFFSET_", weightsBufferOffset);
    REPLACE_WITH_NUM(model, "_WEIGHTS_BYTE_SIZE_", weightsByteSize);

    REPLACE_WITH_NUM(model, "_BIAS_OFFSET_", weightsBufferOffset + weightsByteSize);
    REPLACE_WITH_NUM(model, "_BIAS_BYTE_SIZE_", biasByteSize);

    REPLACE_WITH_NUM(model, "_OUTPUT_BATCH_", input_dims[0]);
    REPLACE_WITH_NUM(model, "_OUTPUT_CHANNEL_", outChannels);

    CNNNetwork network = ie.ReadNetwork(model, weightsBuffer);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);

    ExecutableNetwork executableNetwork;
    ASSERT_NO_THROW(executableNetwork = ie.LoadNetwork(network, "kmb", config));
    ASSERT_NO_THROW(executableNetwork.Export(blob_name));
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

    uint32_t* biasData = reinterpret_cast<uint32_t*>(weightsBuffer->buffer().as<uint8_t*>() + weightsByteSize);
    for (size_t i = 0; i < biasSize; ++i) {
        biasData[i] = 1;
    }

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
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);

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
    {{1, 16, 1, 1}, 16, true, true, ""},
    {{1, 8, 1, 1}, 16, false, true, ""},
    {{1, 16}, 8, false, true, ""},
};

INSTANTIATE_TEST_CASE_P(accuracy, kmbLayersTestsFullyConnectedWithIR, ::testing::ValuesIn(test_params_fc));
#endif
