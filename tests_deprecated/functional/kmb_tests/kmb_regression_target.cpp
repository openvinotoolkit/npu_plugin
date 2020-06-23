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

#include "kmb_regression_target.hpp"

#include <file_reader.h>
#include <gtest/gtest.h>
#include <ie_layers.h>
#include <precision_utils.h>

#include <blob_factory.hpp>
#include <condition_variable>
#include <mutex>
#include <regression_tests.hpp>
#include <test_model/kmb_test_utils.hpp>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu_layers_tests.hpp>
#include <ie_compound_blob.h>

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"
#include "low_precision_transformations/transformer.hpp"
#include "tests_timeout.hpp"
#include "vpu/kmb_params.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;
using namespace TestsTimeout;
using namespace KmbRegressionTarget;

#ifdef ENABLE_MCM_COMPILER
using kmbLayersTestsConvolution = kmbLayersTests_nightly;

TEST_F(kmbLayersTestsConvolution, compilationLoadNetworkAndInfer) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#endif
    std::string model = convolution_u8_only;

    const size_t convolutionWeightsByteSize = 36864;
    const size_t convolutionWeightsSize = convolutionWeightsByteSize / sizeof(uint8_t);

    const size_t biasByteSize = 256;
    const size_t biasSize = biasByteSize / sizeof(int32_t);

    auto convolutionWeightsBuffer =
        make_shared_blob<uint8_t>({Precision::U8, {convolutionWeightsByteSize + biasByteSize}, Layout::C});
    convolutionWeightsBuffer->allocate();
    auto weightsBufferData = convolutionWeightsBuffer->buffer().as<uint8_t*>();
    for (size_t i = 0; i < convolutionWeightsSize; ++i) {
        weightsBufferData[i] = 1;
    }

    uint32_t* biasData =
        reinterpret_cast<uint32_t*>(convolutionWeightsBuffer->buffer().as<uint8_t*>() + convolutionWeightsSize);
    for (size_t i = 0; i < biasSize; ++i) {
        biasData[i] = 1lu;
    }

    CNNNetwork network;
    ASSERT_NO_THROW(network = core->ReadNetwork(model, convolutionWeightsBuffer));

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    Core ie;
    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = core->LoadNetwork(network, deviceName, config));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    std::string input_name = exeNetwork.GetInputsInfo().begin()->first;
    std::string output_name = exeNetwork.GetOutputsInfo().begin()->first;

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(input_name));

    ASSERT_NO_THROW(inferRequest.Infer());

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(output_name));
}
#endif

#if defined(__arm__) || defined(__aarch64__)

const size_t NUMBER_OF_TOP_CLASSES = 5;
const std::string YOLO_GRAPH_NAME = "tiny-yolo-v2.blob";

struct modelBlobsInfo {
    std::string _graphPath, _inputPath, _outputPath;
};

const static std::vector<modelBlobsInfo> pathToPreCompiledGraph = {
    {
        ._graphPath = "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob",
        ._inputPath = "/KMB_models/BLOBS/mobilenet-v2/input.bin",
        ._outputPath = "/KMB_models/BLOBS/mobilenet-v2/output.bin",
    },
    {
        ._graphPath = "/KMB_models/BLOBS/resnet-50/resnet-50.blob",
        ._inputPath = "/KMB_models/BLOBS/resnet-50/input.bin",
        ._outputPath = "/KMB_models/BLOBS/resnet-50/output.bin",
    },
    {
        ._graphPath = "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob",
        ._inputPath = "/KMB_models/BLOBS/tiny-yolo-v2/input.bin",
        ._outputPath = "/KMB_models/BLOBS/tiny-yolo-v2/output.bin",
    }};

class VpuInferWithPath : public vpuLayersTests, public testing::WithParamInterface<modelBlobsInfo> {};

class VpuNoRegressionInference : public Regression::RegressionTests {
public:
    std::string getDeviceName() const override { return ""; }

protected:
    std::string pluginName = "kmbPlugin";
};

TEST_P(VpuInferWithPath, canDoInferenceOnImportedBlob) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string modelFilePath = ModelsPath() + blobsInfo._graphPath;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
    ASSERT_NO_THROW(inferRequest.Infer());
}

TEST_P(VpuInferWithPath, compareInferenceOutputWithReference) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    ConstInputsDataMap inputInfo;
    inputInfo = importedNetwork.GetInputsInfo();

    std::string inputFilePath = ModelsPath() + inputSuffix;
    for (auto& item : inputInfo) {
        Blob::Ptr inputBlob = inferRequest.GetBlob(item.first.c_str());
        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
    }

    ASSERT_NO_THROW(inferRequest.Infer());

    ConstOutputsDataMap outputInfo;
    outputInfo = importedNetwork.GetOutputsInfo();

    for (auto& item : outputInfo) {
        Blob::Ptr outputBlob = inferRequest.GetBlob(item.first.c_str());

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
        Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
            outputBlobTensorDesc.getPrecision(), outputBlobTensorDesc.getDims(), outputBlobTensorDesc.getLayout()));
        referenceOutputBlob->allocate();

        std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));
        Blob::Ptr refFP32 = toFP32(referenceOutputBlob);
        Blob::Ptr outputFP32 = toFP32(outputBlob);
        if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
            Compare(refFP32, outputFP32, 0.0f);
        } else {
            ASSERT_NO_THROW(compareTopClasses(outputFP32, refFP32, NUMBER_OF_TOP_CLASSES));
        }
    }
}

class VpuInferAndCompareTestsWithParam :
    public vpuLayersTests,
    public testing::WithParamInterface<std::tuple<bool, modelBlobsInfo>> {};

TEST_P(VpuInferAndCompareTestsWithParam, multipleInferRequests) {
    std::tuple<bool, modelBlobsInfo> paramTuple = GetParam();
    bool isSync = std::get<0>(paramTuple);
    modelBlobsInfo blobsInfo = std::get<1>(paramTuple);
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    std::list<InferenceEngine::InferRequest> requestList;
    const int REQUEST_LIMIT = 10;
    for (int requestCount = 0; requestCount < REQUEST_LIMIT; requestCount++) {
        InferenceEngine::InferRequest inferRequest;
        ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
        requestList.push_back(inferRequest);
    }

    std::string inputFilePath = ModelsPath() + inputSuffix;

    ConstInputsDataMap inputInfo;
    inputInfo = importedNetwork.GetInputsInfo();

    for (auto& item : inputInfo) {
        for (auto currentRequest : requestList) {
            Blob::Ptr inputBlob;
            inputBlob = currentRequest.GetBlob(item.first.c_str());
            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
        }
    }

    const int MAX_WAIT = 60000;
    auto requestRoutine = [MAX_WAIT](InferenceEngine::InferRequest request) -> void {
        ResponseDesc response;
        ASSERT_NO_THROW(request.StartAsync());
        ASSERT_EQ(StatusCode::OK, request.Wait(MAX_WAIT)) << response.msg;
    };

    if (isSync) {
        // synchronous execution
        for (InferenceEngine::InferRequest& currentRequest : requestList) {
            ASSERT_NO_THROW(currentRequest.Infer());
        }
    } else {
        // asynchronous execution
        std::list<std::thread> requestThreadList;
        for (InferenceEngine::InferRequest& currentRequest : requestList) {
            requestThreadList.push_back(std::thread(requestRoutine, currentRequest));
        }

        for (std::thread& requestThread : requestThreadList) {
            requestThread.join();
        }
    }

    ConstOutputsDataMap outputInfo;
    outputInfo = importedNetwork.GetOutputsInfo();

    std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
    for (auto& item : outputInfo) {
        for (InferenceEngine::InferRequest& inferRequest : requestList) {
            Blob::Ptr outputBlob;
            outputBlob = inferRequest.GetBlob(item.first.c_str());

            TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
            Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
                outputBlobTensorDesc.getPrecision(), outputBlobTensorDesc.getDims(), outputBlobTensorDesc.getLayout()));
            referenceOutputBlob->allocate();

            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));
            Blob::Ptr refFP32 = toFP32(referenceOutputBlob);
            Blob::Ptr outputFP32 = toFP32(outputBlob);
            if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
                Compare(refFP32, outputFP32, 0.0f);
            } else {
                ASSERT_NO_THROW(compareTopClasses(outputFP32, refFP32, NUMBER_OF_TOP_CLASSES));
            }
        }
    }
}

TEST_P(VpuInferWithPath, asyncInferCallback) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    std::list<InferenceEngine::InferRequest> requestList;
    const int REQUEST_LIMIT = 10;
    for (int requestCount = 0; requestCount < REQUEST_LIMIT; requestCount++) {
        InferenceEngine::InferRequest inferRequest;
        ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
        requestList.push_back(inferRequest);
    }

    std::string inputFilePath = ModelsPath() + inputSuffix;

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    for (auto& item : inputInfo) {
        for (auto currentRequest : requestList) {
            Blob::Ptr inputBlob;
            ASSERT_NO_THROW(inputBlob = currentRequest.GetBlob(item.first.c_str()));
            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
        }
    }

    std::mutex requestCounterGuard;
    volatile int completedRequests = 0;
    auto onComplete = [&completedRequests, &requestCounterGuard](void) -> void {
        std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
        completedRequests++;
    };

    // asynchronous execution
    for (InferenceEngine::InferRequest& currentRequest : requestList) {
        currentRequest.SetCompletionCallback(onComplete);
        ASSERT_NO_THROW(currentRequest.StartAsync());
    }

    const int MAX_WAIT = 60000;
    auto waitRoutine = [&completedRequests, MAX_WAIT, REQUEST_LIMIT](void) -> void {
        std::chrono::system_clock::time_point endTime =
            std::chrono::system_clock::now() + std::chrono::milliseconds(MAX_WAIT);

        while (completedRequests < REQUEST_LIMIT) {
            ASSERT_LE(std::chrono::system_clock::now(), endTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    };

    std::thread waitThread(waitRoutine);
    waitThread.join();

    ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();

    std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
    for (auto& item : outputInfo) {
        for (InferenceEngine::InferRequest& inferRequest : requestList) {
            Blob::Ptr outputBlob;
            ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(item.first.c_str()));

            TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
            Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
                outputBlobTensorDesc.getPrecision(), outputBlobTensorDesc.getDims(), outputBlobTensorDesc.getLayout()));
            referenceOutputBlob->allocate();

            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));
            Blob::Ptr refFP32 = toFP32(referenceOutputBlob);
            Blob::Ptr outputFP32 = toFP32(outputBlob);
            if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
                Compare(refFP32, outputFP32, 0.0f);
            } else {
                ASSERT_NO_THROW(compareTopClasses(outputFP32, refFP32, NUMBER_OF_TOP_CLASSES));
            }
        }
    }
}

TEST_P(VpuInferWithPath, asyncInferCallbackRecursive) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    std::string inputFilePath = ModelsPath() + inputSuffix;

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    for (auto& item : inputInfo) {
        Blob::Ptr inputBlob;
        ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(item.first.c_str()));
        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
    }

    const std::size_t MAX_ITERATIONS = 10;
    volatile std::size_t iterationCount = 0;
    std::condition_variable waitCompletion;

    auto onComplete = [&waitCompletion, &iterationCount, &inferRequest](void) -> void {
        iterationCount++;
        if (iterationCount < MAX_ITERATIONS) {
            ASSERT_NO_THROW(inferRequest.StartAsync());
        } else {
            waitCompletion.notify_one();
        }
    };

    inferRequest.SetCompletionCallback(onComplete);

    ASSERT_NO_THROW(inferRequest.StartAsync());

    std::mutex execGuard;
    std::unique_lock<std::mutex> execLocker(execGuard);
    waitCompletion.wait(execLocker, [&] {
        return iterationCount == MAX_ITERATIONS;
    });

    ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();

    std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
    for (auto& item : outputInfo) {
        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(item.first.c_str()));

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
        Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
            outputBlobTensorDesc.getPrecision(), outputBlobTensorDesc.getDims(), outputBlobTensorDesc.getLayout()));
        referenceOutputBlob->allocate();

        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

        Blob::Ptr refFP32 = toFP32(referenceOutputBlob);
        Blob::Ptr outputFP32 = toFP32(outputBlob);
        if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
            Compare(refFP32, outputFP32, 0.0f);
        } else {
            ASSERT_NO_THROW(compareTopClasses(outputFP32, refFP32, NUMBER_OF_TOP_CLASSES));
        }
    }
}

const static std::vector<bool> isSyncVec = {false, true};

INSTANTIATE_TEST_CASE_P(multipleInference, VpuInferAndCompareTestsWithParam,
    ::testing::Combine(::testing::ValuesIn(isSyncVec), ::testing::ValuesIn(pathToPreCompiledGraph)));

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
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputBlob1));

    ASSERT_NO_THROW(inferRequest1.Infer());
    std::string output_name1 = importedNetwork.GetOutputsInfo().begin()->first;
    Blob::Ptr outputBlob1;
    ASSERT_NO_THROW(outputBlob1 = inferRequest1.GetBlob(output_name1));

    // ----------------------------------------------------------

    InferenceEngine::TensorDesc inputTensorDesc = inferRequest1.GetBlob(input_name1)->getTensorDesc();
    InferenceEngine::TensorDesc outputTensorDesc = inferRequest1.GetBlob(output_name1)->getTensorDesc();

    Blob::Ptr fileOutputBlob =
        InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, outputTensorDesc.getDims(), Layout::NCHW});
    fileOutputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, fileOutputBlob));

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = InferenceEngine::make_shared_blob<uint8_t>(
                        {Precision::U8, inputTensorDesc.getDims(), Layout::NCHW}));
    inputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputBlob));

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
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, mobilNInputBlob));

    std::string resNInput_name = resNImportedNetwork.GetInputsInfo().begin()->first;
    ASSERT_NO_THROW(resNInferRequest.SetBlob(resNInput_name, mobilNInputBlob));
    Blob::Ptr resNInputBlob;
    ASSERT_NO_THROW(resNInputBlob = resNInferRequest.GetBlob(resNInput_name));
    ASSERT_EQ((void*)mobilNInputBlob->buffer(), (void*)resNInputBlob->buffer());
}

// [Track number: S#27180]
TEST_P(VpuInferWithPath, DISABLED_compareOutputsTwoNetworks) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;
    std::string inputNameFilePath = ModelsPath() + inputSuffix;
    std::string outputNameFilePath = ModelsPath() + outputSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork1;
    ASSERT_NO_THROW(importedNetwork1 = core->ImportNetwork(modelFilePath, deviceName));
    InferenceEngine::ExecutableNetwork importedNetwork2;
    ASSERT_NO_THROW(importedNetwork2 = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest1;
    ASSERT_NO_THROW(inferRequest1 = importedNetwork1.CreateInferRequest());

    std::string input_name1 = importedNetwork1.GetInputsInfo().begin()->first;
    std::string output_name1 = importedNetwork1.GetOutputsInfo().begin()->first;

    InferenceEngine::TensorDesc outputTensorDesc = inferRequest1.GetBlob(output_name1)->getTensorDesc();

    Blob::Ptr fileOutputBlob =
        InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, outputTensorDesc.getDims(), Layout::NCHW});
    fileOutputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, fileOutputBlob));

    Blob::Ptr inputBlob1;
    ASSERT_NO_THROW(inputBlob1 = inferRequest1.GetBlob(input_name1));
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputBlob1));

    ASSERT_NO_THROW(inferRequest1.Infer());
    Blob::Ptr outputBlob1;
    ASSERT_NO_THROW(outputBlob1 = inferRequest1.GetBlob(output_name1));

    // --------------------

    InferenceEngine::InferRequest InferRequest2;
    ASSERT_NO_THROW(InferRequest2 = importedNetwork2.CreateInferRequest());

    std::string input_name2 = importedNetwork2.GetInputsInfo().begin()->first;
    std::string output_name2 = importedNetwork2.GetOutputsInfo().begin()->first;

    ASSERT_NO_THROW(InferRequest2.SetBlob(input_name2, inputBlob1));
    ASSERT_NO_THROW(InferRequest2.Infer());
    Blob::Ptr outputBlob2;
    ASSERT_NO_THROW(outputBlob2 = InferRequest2.GetBlob(output_name2));
    ASSERT_EQ(outputBlob1->byteSize(), outputBlob2->byteSize());

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

    Blob::Ptr fileOutputBlob =
        InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, outputTensorDesc.getDims(), Layout::NCHW});
    fileOutputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, fileOutputBlob));

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = InferenceEngine::make_shared_blob<uint8_t>(
                        {Precision::U8, inputTensorDesc.getDims(), Layout::NCHW}));
    inputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputBlob));

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

TEST_F(vpuInferWithSetUp, copyCheckSetBlob) {
    std::string strToCheck = "isValidPtr(): Input blob will be copied";
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
    std::string inputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
    std::string outputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output.bin";

    Blob::Ptr fileOutputBlob = InferenceEngine::make_shared_blob<ie_fp16>({Precision::FP16, {1, 1000}, Layout::NC});
    fileOutputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, fileOutputBlob));

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
    std::string input_name = importedNetwork.GetInputsInfo().begin()->first;

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(input_name));
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputNameFilePath, inputBlob));
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

struct TopNetTest {
    modelBlobsInfo info;
    bool isComparable;
};

const static std::vector<TopNetTest> pathToTop3PreCompiledGraph = {
    {{
         ._graphPath = "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob",
         ._inputPath = "/KMB_models/BLOBS/mobilenet-v2/input.bin",
         ._outputPath = "/KMB_models/BLOBS/mobilenet-v2/output.bin",
     },
        true},
    {{
         ._graphPath = "/KMB_models/BLOBS/resnet-50/resnet-50.blob",
         ._inputPath = "/KMB_models/BLOBS/resnet-50/input.bin",
         ._outputPath = "/KMB_models/BLOBS/resnet-50/output.bin",
     },
        true},
    {{
         ._graphPath = "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob",
         ._inputPath = "/KMB_models/BLOBS/tiny-yolo-v2/input.bin",
         ._outputPath = "/KMB_models/BLOBS/tiny-yolo-v2/output.bin",
     },
        false}};

class VpuInferWithPathForTop3Net : public vpuLayersTests, public testing::WithParamInterface<TopNetTest> {};

TEST_P(VpuInferWithPathForTop3Net, canDoInferenceOnTop3ImportedBlobs) {
    modelBlobsInfo blobsInfo = GetParam().info;
    std::string modelFilePath = ModelsPath() + blobsInfo._graphPath;
    std::string inputDataPath = ModelsPath() + blobsInfo._inputPath;
    std::string refOutputPath = ModelsPath() + blobsInfo._outputPath;

    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
    auto inputBlobName = importedNetwork.GetInputsInfo().begin()->first;
    auto inputBlob = inferRequest.GetBlob(inputBlobName);
    vpu::KmbPlugin::utils::fromBinaryFile(inputDataPath, inputBlob);

    ASSERT_NO_THROW(inferRequest.Infer());

    if (!GetParam().isComparable) return;

    auto outputBlobName = importedNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    auto refBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    refBlob->allocate();
    vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, refBlob);

    ASSERT_TRUE(outputBlob->byteSize() == refBlob->byteSize());
    ASSERT_TRUE(outputBlob->getTensorDesc().getPrecision() == Precision::FP16 ||
                outputBlob->getTensorDesc().getPrecision() == Precision::FP32);
    ASSERT_NO_THROW(compareTopClasses(toFP32(outputBlob), toFP32(refBlob), NUMBER_OF_TOP_CLASSES));
}

INSTANTIATE_TEST_CASE_P(inferenceWithParameters, VpuInferWithPath, ::testing::ValuesIn(pathToPreCompiledGraph));

INSTANTIATE_TEST_CASE_P(
    inferenceWithTop3Networks, VpuInferWithPathForTop3Net, ::testing::ValuesIn(pathToTop3PreCompiledGraph));

TEST_P(VpuInferWithPathForTop3Net, remoteCtx) {
    const modelBlobsInfo blobsInfo = GetParam().info;
    const std::string graphPath = ModelsPath() + blobsInfo._graphPath;
    const std::string refInputPath = ModelsPath() + blobsInfo._inputPath;
    const std::string refOutputPath = ModelsPath() + blobsInfo._outputPath;

    const ParamMap ctxParams = {{"LOG_LEVEL", "LOG_DEBUG"}};
    InferenceEngine::Core ie;
    InferenceEngine::RemoteContext::Ptr contextPtr = ie.CreateContext("KMB", ctxParams);

    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);

    const std::map<std::string, std::string> netParams = {{"LOG_LEVEL", "LOG_DEBUG"}};
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, netParams);
    InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    InferenceEngine::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    const TensorDesc& inputTensorDesc = inputInfoPtr->getTensorDesc();
    const size_t bytesPerElement = inputTensorDesc.getPrecision().size();
    auto binaryMul = [](const size_t& multiplier, const size_t& multiplicand) -> size_t {
        return multiplier * multiplicand;
    };
    auto vpuAllocator = std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    size_t imageSize = std::accumulate(
        inputTensorDesc.getDims().begin(), inputTensorDesc.getDims().end(), bytesPerElement, binaryMul);
    void* virtAddr = vpuAllocator->allocate(imageSize);
    auto remoteMemoryFd = vpuAllocator->getFileDescByVirtAddr(virtAddr);
    InferenceEngine::ParamMap blobParamMap = {
        {InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd},
        {InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE), virtAddr},
    };

    InferenceEngine::RemoteBlob::Ptr remoteBlobPtr = contextPtr->CreateBlob(inputTensorDesc, blobParamMap);
    ASSERT_NE(nullptr, remoteBlobPtr);

    InferenceEngine::MemoryBlob::Ptr userBlob = make_shared_blob<uint8_t>(inputTensorDesc);
    userBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refInputPath, userBlob));
    std::memcpy(virtAddr, userBlob->rmap(), userBlob->byteSize());

    inferRequest.SetBlob(inputName, remoteBlobPtr);
    inferRequest.Infer();

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto outputRefBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    outputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputRefBlob));

    // --- Compare with expected output
    constexpr size_t numberOfTopClassesToCompare = 3;
    ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(outputRefBlob), numberOfTopClassesToCompare));
}

TEST_F(vpuLayersTests, remoteCtxNV12) {
    const modelBlobsInfo blobsInfo = pathToPreCompiledGraph.at(1);
    const std::string graphPath = ModelsPath() + blobsInfo._graphPath;
    const std::string refInputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-dog-1080x1080-nv12.bin";
    const std::string refOutputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-dog-1080x1080-nv12.bin";

    const ParamMap ctxParams = {{"LOG_LEVEL", "LOG_DEBUG"}};
    InferenceEngine::Core ie;
    InferenceEngine::RemoteContext::Ptr contextPtr = ie.CreateContext("KMB", ctxParams);

    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);

    const std::map<std::string, std::string> netParams = {{"LOG_LEVEL", "LOG_DEBUG"}};
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, netParams);
    InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    InferenceEngine::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> vpuAllocator = std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    InferenceEngine::Blob::Ptr remoteBlobPtr;

    const size_t imWidth = 1080;
    const size_t imHeight = 1080;
    InferenceEngine::Blob::Ptr userBlob =
        vpu::KmbPlugin::utils::fromNV12File(refInputPath, imWidth, imHeight, vpuAllocator);
    NV12Blob::Ptr userNV12blobPtr = as<NV12Blob>(userBlob);
    MemoryBlob::Ptr userYPlane = as<MemoryBlob>(userNV12blobPtr->y());
    MemoryBlob::Ptr userUVPlane = as<MemoryBlob>(userNV12blobPtr->uv());
    auto remoteMemoryFd = vpuAllocator->getFileDescByVirtAddr(userYPlane->rmap().as<void*>());

    TensorDesc ydesc(Precision::U8, { 1, 1, imHeight, imWidth }, Layout::NHWC);
    ParamMap blobParams = {
        { InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFd },
        { InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE), userYPlane->rmap().as<KmbHandleParam>() },
    };
    Blob::Ptr y_blob = std::dynamic_pointer_cast<Blob>(contextPtr->CreateBlob(ydesc, blobParams));

    TensorDesc uvdesc(Precision::U8, { 1, 2, imHeight / 2, imWidth / 2 }, Layout::NHWC);
    blobParams[InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE)] = userUVPlane->rmap().as<KmbHandleParam>();
    Blob::Ptr uv_blob = std::dynamic_pointer_cast<Blob>(contextPtr->CreateBlob(uvdesc, blobParams));

    remoteBlobPtr = make_shared_blob<NV12Blob>(y_blob, uv_blob);

    PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
    preprocInfo.setColorFormat(ColorFormat::NV12);

    inferRequest.SetBlob(inputName, remoteBlobPtr, preprocInfo);
    inferRequest.Infer();

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto outputRefBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    outputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputRefBlob));

    // --- Compare with expected output
    constexpr size_t numberOfTopClassesToCompare = 3;
    ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(outputRefBlob), numberOfTopClassesToCompare));
}

struct rectangle {
    size_t x;
    size_t y;
    size_t width;
    size_t height;
};

struct Image {
    size_t width;
    size_t height;
    rectangle rect;
    uint8_t* planes[2];
    int remoteMemoryFd[2];
};

static Blob::Ptr wrapImageToBlob(const Image& image, const RemoteContext::Ptr& contextPtr) {
    const size_t& imageWidth = image.width;
    const size_t& imageHeight = image.height;
    TensorDesc planeY(Precision::U8, {1, 1, imageHeight, imageWidth}, Layout::NHWC);
    TensorDesc planeUV(Precision::U8, {1, 2, imageHeight / 2, imageWidth / 2}, Layout::NHWC);
    ROI crop_roi_y({0, image.rect.x, image.rect.y, image.rect.width, image.rect.height});
    ROI crop_roi_uv({0, image.rect.x / 2, image.rect.y / 2, image.rect.width / 2, image.rect.height / 2});

    ParamMap paramsY = {
        { InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), image.remoteMemoryFd[0] },
        { InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE), reinterpret_cast<void*>(image.planes[0]) },
    };
    RemoteBlob::Ptr blobY = contextPtr->CreateBlob(planeY, paramsY);
    if (blobY == nullptr) {
        throw std::runtime_error("Failed to create remote blob for Y plane");
    }

    ParamMap paramsUV = {
        { InferenceEngine::KMB_PARAM_KEY(REMOTE_MEMORY_FD), image.remoteMemoryFd[1] },
        { InferenceEngine::KMB_PARAM_KEY(MEM_HANDLE), reinterpret_cast<void*>(image.planes[1]) },
    };
    RemoteBlob::Ptr blobUV = contextPtr->CreateBlob(planeUV, paramsUV);
    if (blobY == nullptr) {
        throw std::runtime_error("Failed to create remote blob for UV plane");
    }

    Blob::Ptr y_plane_with_roi = make_shared_blob(blobY, crop_roi_y);
    Blob::Ptr uv_plane_with_roi = make_shared_blob(blobUV, crop_roi_uv);

    Blob::Ptr nv12Blob = make_shared_blob<NV12Blob>(y_plane_with_roi, uv_plane_with_roi);
    return nv12Blob;
}

TEST_F(vpuLayersTests, remoteCtxNV12WithROI) {
    const modelBlobsInfo blobsInfo = pathToPreCompiledGraph.at(1);
    const std::string graphPath = ModelsPath() + blobsInfo._graphPath;
    const std::string refInputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-dog-1080x1080-nv12.bin";
    const std::string refOutputPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-dog-1080x1080-nv12.bin";

    const ParamMap ctxParams = {{"LOG_LEVEL", "LOG_DEBUG"}};
    InferenceEngine::Core ie;
    InferenceEngine::RemoteContext::Ptr contextPtr = ie.CreateContext("KMB", ctxParams);

    std::filebuf blobFile;
    if (!blobFile.open(graphPath, std::ios::in | std::ios::binary)) {
        blobFile.close();
        THROW_IE_EXCEPTION << "Could not open file: " << graphPath;
    }
    std::istream graphBlob(&blobFile);

    const std::map<std::string, std::string> netParams = {{"LOG_LEVEL", "LOG_DEBUG"}};
    InferenceEngine::ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, netParams);
    InferenceEngine::InferRequest inferRequest = executableNetwork.CreateInferRequest();

    auto inputsInfo = executableNetwork.GetInputsInfo();
    const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;
    InferenceEngine::InputInfo::CPtr inputInfoPtr = executableNetwork.GetInputsInfo().begin()->second;

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> vpuAllocator = std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    const size_t imWidth = 1080;
    const size_t imHeight = 1080;
    const size_t lumaSize = imWidth * imHeight;
    const size_t yuvSize = lumaSize * 3 / 2;
    auto imageBaseAddr = reinterpret_cast<uint8_t*>(vpuAllocator->allocate(yuvSize));
    auto remoteMemoryFd = vpuAllocator->getFileDescByVirtAddr(imageBaseAddr);
    std::ifstream inputFileHandle(refInputPath, std::ios_base::binary);
    inputFileHandle.read(reinterpret_cast<char*>(imageBaseAddr), yuvSize);
    inputFileHandle.close();
    Image frame;
    frame.width = imWidth,
    frame.height = imHeight,
    frame.rect.x = 0;
    frame.rect.y = 0;
    frame.rect.width = imWidth;
    frame.rect.height = imHeight;
    frame.planes[0] = imageBaseAddr,
    frame.planes[1] = imageBaseAddr + lumaSize,
    frame.remoteMemoryFd[0] = remoteMemoryFd;
    frame.remoteMemoryFd[1] = remoteMemoryFd;
    InferenceEngine::Blob::Ptr remoteBlobPtr = wrapImageToBlob(frame, contextPtr);

    PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
    preprocInfo.setColorFormat(ColorFormat::NV12);

    inferRequest.SetBlob(inputName, remoteBlobPtr, preprocInfo);
    inferRequest.Infer();

    // --- Get output
    auto outputBlobName = executableNetwork.GetOutputsInfo().begin()->first;
    auto outputBlob = inferRequest.GetBlob(outputBlobName);

    // --- Reference Blob
    auto outputRefBlob = make_blob_with_precision(outputBlob->getTensorDesc());
    outputRefBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(refOutputPath, outputRefBlob));

    // --- Compare with expected output
    constexpr size_t numberOfTopClassesToCompare = 3;
    ASSERT_NO_THROW(Comparators::compareTopClasses(toFP32(outputBlob), toFP32(outputRefBlob), numberOfTopClassesToCompare));
}
#endif
