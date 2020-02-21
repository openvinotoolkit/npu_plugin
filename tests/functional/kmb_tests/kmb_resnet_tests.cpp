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

#include <file_reader.h>

#include <blob_factory.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <conv_ref.hpp>
#include <ie_icnn_network_stats.hpp>
#include <ie_util_internal.hpp>
#include <vpu/kmb_plugin_config.hpp>

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace details;

struct resnet_params {
    std::string modelPath, weightsPath, inputPath, exepectedResPath;
    float scale;
    uint8_t shift;
};

class ResnetTest : public kmbLayersTests_nightly, public testing::WithParamInterface<resnet_params> {};

Blob::Ptr dequantize(const Blob::Ptr& blobIn, float scale, uint8_t shift) {
    Blob::Ptr blobOut = make_blob_with_precision(TensorDesc(
        InferenceEngine::Precision::FP32, blobIn->getTensorDesc().getDims(), blobIn->getTensorDesc().getLayout()));
    blobOut->allocate();
    const uint8_t* rawBufferIn = blobIn->cbuffer().as<uint8_t*>();
    float* rawBufferOut = blobOut->buffer().as<float*>();
    for (size_t byteIdx = 0; byteIdx < blobIn->size(); byteIdx++) {
        rawBufferOut[byteIdx] = (rawBufferIn[byteIdx] - shift) * scale;
    }
    return blobOut;
}

#ifdef ENABLE_MCM_COMPILER

// [Track number: S#27240]
TEST_P(ResnetTest, DISABLED_resnetAccuracy) {
#ifndef ENABLE_VPUAL
    SKIP();
#endif
    resnet_params test_params = GetParam();
    std::string fullPathToModelXML = ModelsPath() + "/KMB_models/resnet50/" + test_params.modelPath;
    std::string fullPathToWeights = ModelsPath() + "/KMB_models/resnet50/" + test_params.weightsPath;

    Core ie;
    CNNNetwork network = ie.ReadNetwork(fullPathToModelXML, fullPathToWeights);

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo.begin()->second->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo.begin()->second->setPrecision(Precision::U8);
    _outputsInfo.begin()->second->setLayout(Layout::NHWC);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    InferenceEngine::ExecutableNetwork exeNetwork;
    exeNetwork = ie.LoadNetwork(network, "KMB", config);

    InferenceEngine::InferRequest inferRequest;
    inferRequest = exeNetwork.CreateInferRequest();

    Blob::Ptr inputBlob = inferRequest.GetBlob(exeNetwork.GetInputsInfo().begin()->first);
    std::string inputFilePath = ModelsPath() + "/KMB_models/resnet50/" + test_params.inputPath;
    vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob);

    inferRequest.Infer();

    ConstOutputsDataMap outputInfo;
    outputInfo = exeNetwork.GetOutputsInfo();

    for (auto& item : outputInfo) {
        Blob::Ptr outputBlob = inferRequest.GetBlob(item.first.c_str());

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
        Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
            InferenceEngine::Precision::FP32, outputBlobTensorDesc.getDims(), outputBlobTensorDesc.getLayout()));
        referenceOutputBlob->allocate();

        std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/resnet50/" + test_params.exepectedResPath;
        vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob);

        float scale = test_params.scale;
        uint8_t shift = test_params.shift;
        Blob::Ptr outputFP32 = dequantize(outputBlob, scale, shift);
        Compare(referenceOutputBlob, outputFP32, 0.0f);
    }
}

static const std::vector<resnet_params> resnetTestParams = {
    {
        "scale-shift.xml",                        // model
        "resnet50-weights.bin",                   // weights
        "input-cat-224x224-interleaved.bgr.bin",  // input
        "scale-shift-reference.bin",              // expected result
        0.0197619f,
        128,
    },
    {
        "single-convolution.xml",                 // model
        "resnet50-weights.bin",                   // weights
        "input-cat-224x224-interleaved.bgr.bin",  // input
        "single-convolution-reference.bin",       // expected result
        0.0127005f,
        0,
    },
};

static const std::vector<resnet_params> resnetTestParamsFail = {
    {
        "first-block.xml",                        // model
        "resnet50-weights.bin",                   // weights
        "input-cat-224x224-interleaved.bgr.bin",  // input
        "first-block-reference.bin",              // expected result
        0.00880455f,
        0,
    },
    {
        "three-blocks.xml",                       // model
        "resnet50-weights.bin",                   // weights
        "input-cat-224x224-interleaved.bgr.bin",  // input
        "three-blocks-reference.bin",             // expected result
        0.0106719f,
        0,
    },
};

INSTANTIATE_TEST_CASE_P(resnetAccuracyTests, ResnetTest, ::testing::ValuesIn(resnetTestParams));
// [Track number: S#27240]
INSTANTIATE_TEST_CASE_P(DISABLED_resnetAccuracyTestsFail, ResnetTest, ::testing::ValuesIn(resnetTestParamsFail));

#endif
