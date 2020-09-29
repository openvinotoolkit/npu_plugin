//
// Copyright 2020 Intel Corporation.
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

#include "kmb_regression_target.hpp"
#include <vpu_layers_tests.hpp>
#include <gtest/gtest.h>
#include <file_reader.h>
#include <test_model/kmb_test_utils.hpp>
#include <regression_tests.hpp>

using namespace KmbRegressionTarget;

#if defined(__arm__) || defined(__aarch64__)

TEST_P(VpuInferWithPath, compareSetBlobAndGetBlobInput) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();
    std::string input_name = inputInfo.begin()->first;

    InferenceEngine::TensorDesc inputTensorDesc = inferRequest.GetBlob(input_name)->getTensorDesc();

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = InferenceEngine::make_shared_blob<uint8_t>(
                        {Precision::U8, inputTensorDesc.getDims(), Layout::NCHW}));
    inputBlob->allocate();

    auto inputBufferData = inputBlob->buffer().as<uint8_t*>();
    std::fill(inputBufferData, inputBufferData + inputBlob->byteSize(), 0xFF);

    ASSERT_NO_THROW(inferRequest.SetBlob(input_name, inputBlob));
    Blob::Ptr newInputBlob;
    ASSERT_NO_THROW(newInputBlob = inferRequest.GetBlob(input_name));

    ASSERT_TRUE(std::equal(static_cast<u_int8_t*>(inputBlob->buffer()),
        static_cast<u_int8_t*>(inputBlob->buffer()) + inputBlob->byteSize(),
        static_cast<u_int8_t*>(newInputBlob->buffer())));
    ASSERT_EQ((void*)inputBlob->buffer(), (void*)newInputBlob->buffer());
}

TEST_P(VpuInferWithPath, compareSetBlobAndGetBlobOutput) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();

    ASSERT_TRUE(!outputInfo.empty());

    std::string output_name = outputInfo.begin()->first;

    InferenceEngine::TensorDesc outputTensorDesc = inferRequest.GetBlob(output_name)->getTensorDesc();

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = InferenceEngine::make_shared_blob<float>(outputTensorDesc));
    outputBlob->allocate();

    auto outputBufferData = outputBlob->buffer().as<uint8_t*>();
    std::fill(outputBufferData, outputBufferData + outputBlob->byteSize(), 0xFF);

    ASSERT_NO_THROW(inferRequest.SetBlob(output_name, outputBlob));
    Blob::Ptr newOutputBlob;
    ASSERT_NO_THROW(newOutputBlob = inferRequest.GetBlob(output_name));

    ASSERT_TRUE(std::equal(static_cast<u_int8_t*>(outputBlob->buffer()),
        static_cast<u_int8_t*>(outputBlob->buffer()) + outputBlob->byteSize(),
        static_cast<u_int8_t*>(newOutputBlob->buffer())));
    ASSERT_EQ((void*)outputBlob->buffer(), (void*)newOutputBlob->buffer());
}

// [Track number: S#27179]
TEST_P(VpuInferWithPath, DISABLED_compareSetBlobAndGetBlobAfterInfer) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;
    std::string inputNameFilePath = ModelsPath() + inputSuffix;
    std::string outputNameFilePath = ModelsPath() + outputSuffix;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest1;
    ASSERT_NO_THROW(inferRequest1 = importedNetwork.CreateInferRequest());
    std::string input_name1 = importedNetwork.GetInputsInfo().begin()->first;

    Blob::Ptr inputBlob1;
    ASSERT_NO_THROW(inputBlob1 = inferRequest1.GetBlob(input_name1));
    ASSERT_NO_THROW(inputBlob1 = vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputBlob1->getTensorDesc()));

    ASSERT_NO_THROW(inferRequest1.Infer());
    std::string output_name1 = importedNetwork.GetOutputsInfo().begin()->first;
    Blob::Ptr outputBlob1;
    ASSERT_NO_THROW(outputBlob1 = inferRequest1.GetBlob(output_name1));

    // ----------------------------------------------------------

    InferenceEngine::TensorDesc inputTensorDesc = inferRequest1.GetBlob(input_name1)->getTensorDesc();
    InferenceEngine::TensorDesc outputTensorDesc = inferRequest1.GetBlob(output_name1)->getTensorDesc();

    Blob::Ptr fileOutputBlob;
    ASSERT_NO_THROW(fileOutputBlob = vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, outputTensorDesc));

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputTensorDesc));

    InferenceEngine::InferRequest inferRequest2;
    ASSERT_NO_THROW(inferRequest2 = importedNetwork.CreateInferRequest());
    std::string input_name2 = importedNetwork.GetInputsInfo().begin()->first;
    ASSERT_NO_THROW(inferRequest2.SetBlob(input_name2, inputBlob));

    ASSERT_NO_THROW(inferRequest2.Infer());
    std::string output_name2 = importedNetwork.GetOutputsInfo().begin()->first;
    Blob::Ptr outputBlob2;
    ASSERT_NO_THROW(outputBlob2 = inferRequest2.GetBlob(output_name2));
    if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
        Blob::Ptr blobFP32 = toFP32(outputBlob2);
        Blob::Ptr expectedBlobFP32 = toFP32(outputBlob1);
        Compare(blobFP32, expectedBlobFP32, 0.0f);

        blobFP32 = toFP32(outputBlob2);
        expectedBlobFP32 = toFP32(fileOutputBlob);
        Compare(blobFP32, expectedBlobFP32, 0.0f);
    } else {
        Blob::Ptr outputBlob2FP32 = toFP32(outputBlob2);
        ASSERT_NO_THROW(compareTopClasses(outputBlob2FP32, toFP32(outputBlob1), NUMBER_OF_TOP_CLASSES));
        ASSERT_NO_THROW(compareTopClasses(outputBlob2FP32, toFP32(fileOutputBlob), NUMBER_OF_TOP_CLASSES));
    }
}

using kmbSetBlob = vpuLayersTests;

TEST_F(kmbSetBlob, compareSetBlobAllocation) {
    std::string mobilenetModelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
    std::string resnetModelFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";

    std::string inputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input.bin";

    Core ie;
    InferenceEngine::ExecutableNetwork mobilNImportedNetwork;
    ASSERT_NO_THROW(mobilNImportedNetwork = core->ImportNetwork(mobilenetModelFilePath, deviceName));
    InferenceEngine::ExecutableNetwork resNImportedNetwork;
    ASSERT_NO_THROW(resNImportedNetwork = core->ImportNetwork(resnetModelFilePath, deviceName));

    InferenceEngine::InferRequest mobilNInferRequest;
    ASSERT_NO_THROW(mobilNInferRequest = mobilNImportedNetwork.CreateInferRequest());
    InferenceEngine::InferRequest resNInferRequest;
    ASSERT_NO_THROW(resNInferRequest = resNImportedNetwork.CreateInferRequest());

    std::string mobilInput_name = mobilNImportedNetwork.GetInputsInfo().begin()->first;
    Blob::Ptr mobilNInputBlob;
    ASSERT_NO_THROW(mobilNInputBlob = resNInferRequest.GetBlob(mobilInput_name));
    ASSERT_NO_THROW(
        mobilNInputBlob = vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, mobilNInputBlob->getTensorDesc()));

    std::string resNInput_name = resNImportedNetwork.GetInputsInfo().begin()->first;
    ASSERT_NO_THROW(resNInferRequest.SetBlob(resNInput_name, mobilNInputBlob));
    Blob::Ptr resNInputBlob;
    ASSERT_NO_THROW(resNInputBlob = resNInferRequest.GetBlob(resNInput_name));
    ASSERT_EQ((void*)mobilNInputBlob->buffer(), (void*)resNInputBlob->buffer());
}

// [Track number: S#27181]
TEST_P(VpuInferWithPath, DISABLED_compareSetBlobAndInfer) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;
    std::string inputNameFilePath = ModelsPath() + inputSuffix;
    std::string outputNameFilePath = ModelsPath() + outputSuffix;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    std::string input_name = importedNetwork.GetInputsInfo().begin()->first;
    std::string output_name = importedNetwork.GetOutputsInfo().begin()->first;

    InferenceEngine::TensorDesc inputTensorDesc = inferRequest.GetBlob(input_name)->getTensorDesc();
    InferenceEngine::TensorDesc outputTensorDesc = inferRequest.GetBlob(output_name)->getTensorDesc();

    Blob::Ptr fileOutputBlob;
    ASSERT_NO_THROW(fileOutputBlob = vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, outputTensorDesc));

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputTensorDesc));

    ASSERT_NO_THROW(inferRequest.SetBlob(input_name, inputBlob));
    ASSERT_NO_THROW(inferRequest.Infer());

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(output_name));

    Blob::Ptr expectedOutputFP32 = ConvertU8ToFP32(fileOutputBlob);
    Blob::Ptr outputBlobFP32 = ConvertU8ToFP32(outputBlob);
    Compare(expectedOutputFP32, outputBlobFP32, 0.0f);
    if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
        expectedOutputFP32 = ConvertU8ToFP32(fileOutputBlob);
        outputBlobFP32 = ConvertU8ToFP32(outputBlob);
        Compare(expectedOutputFP32, outputBlobFP32, 0.0f);
    } else {
        ASSERT_NO_THROW(compareTopClasses(outputBlob, fileOutputBlob, NUMBER_OF_TOP_CLASSES));
    }
}

class vpuInferWithSetUp : public vpuLayersTests {
public:
    std::stringstream out;
    Regression::basic_streambuf<char, std::char_traits<char>>* ptr;
    void SetUp() override {
        vpuLayersTests::SetUp();
        ptr = std::cout.rdbuf();
        std::cout.rdbuf(out.rdbuf());
    }

    void TearDown() override {
        vpuLayersTests::TearDown();
        std::cout.rdbuf(ptr);
    }
};

TEST_F(vpuInferWithSetUp, DISABLED_copyCheckSetBlob) {
    std::string strToCheck = "isValidPtr(): Input blob will be copied";
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
    std::string inputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
    std::string outputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output.bin";

    Blob::Ptr fileOutputBlob;
    ASSERT_NO_THROW(fileOutputBlob = vpu::KmbPlugin::utils::fromBinaryFile(
                        outputNameFilePath, {Precision::FP16, {1, 1000}, Layout::NC}));

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
    std::string input_name = importedNetwork.GetInputsInfo().begin()->first;

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(input_name));
    ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputBlob->getTensorDesc()));
    ASSERT_NO_THROW(inferRequest.SetBlob(input_name, inputBlob));

    ASSERT_NO_THROW(inferRequest.Infer());
    std::string output_name = importedNetwork.GetOutputsInfo().begin()->first;
    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(output_name));

    Blob::Ptr blobFP32 = toFP32(outputBlob);
    Blob::Ptr expectedBlobFP32 = toFP32(fileOutputBlob);
    Compare(blobFP32, expectedBlobFP32, 0.0f);

    ASSERT_TRUE(out.str().find(strToCheck) == std::string::npos);
}

#endif
