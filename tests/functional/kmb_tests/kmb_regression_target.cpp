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

#include "tests_timeout.hpp"

#include <gtest/gtest.h>
#include <regression_tests.hpp>
#include <inference_engine/precision_utils.h>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>

#include <vpu_layers_tests.hpp>

#include <mutex>
#include <condition_variable>

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;
using namespace TestsTimeout;

namespace
{

class CompilationParameter {
public:
    CompilationParameter() = default;
    inline CompilationParameter(std::string name,
                                std::string path_to_network,
                                std::string path_to_weights);
    //Accessors

    inline std::string name() const;
    inline std::string pathToNetwork() const;
    inline std::string pathToWeights() const;

protected:
    //Data section
    std::string name_;
    std::string path_to_network_;
    std::string path_to_weights_;
};

using Compile = bool;
using Timeout = double;
using CompilationTestParam = WithParamInterface<std::tuple<std::string, CompilationParameter, Compile, Timeout>>;

class VpuNoRegressionWithCompilation : public Regression::RegressionTests,
                                       public CompilationTestParam {
public:
    using TestParam = WithParamInterface<std::tuple<std::string, CompilationParameter, Compile, Timeout>>;

    // Operations
    static std::string getTestCaseName(TestParamInfo <CompilationTestParam::ParamType> param);
    inline void loadNetworkWrapper(std::map<std::string, std::string> config, InferenceEngine::StatusCode* st = nullptr);

    // Accessors
    std::string getPluginName() const override ;
    std::string getDeviceName() const override ;
    std::map<std::string, std::string> _config;

protected:
    // Data section
    std::string pluginName;
    CompilationParameter path_to_files;

    //Operations
    void SetUp() override;
};

std::string VpuNoRegressionWithCompilation::getTestCaseName(
        TestParamInfo <CompilationTestParam::ParamType> param) {
    return std::string("Main_") +
           get<0>(param.param) +
           std::string("_") + get<1>(param.param).name() +
           ((get<2>(param.param)) ? std::string("_Compilation") : std::string("_Parsing"));
}

void VpuNoRegressionWithCompilation::SetUp() {
    pluginName = get<0>(TestParam::GetParam());
    path_to_files = get<1>(TestParam::GetParam());

    PluginCache::get().reset();
}

inline std::string VpuNoRegressionWithCompilation::getPluginName() const {
    return pluginName;
}

inline std::string VpuNoRegressionWithCompilation::getDeviceName() const {
    return "";
}

class KmbNoRegressionCompilationOnly : public VpuNoRegressionWithCompilation {
};

inline CompilationParameter::CompilationParameter(
        std::string name,
        std::string path_to_network,
        std::string path_to_weights):
        name_(name),
        path_to_network_(path_to_network),
        path_to_weights_(path_to_weights)
{
}

inline std::string CompilationParameter::name() const {
    return name_;
}

inline std::string CompilationParameter::pathToNetwork() const {
    return path_to_network_;
}

inline std::string CompilationParameter::pathToWeights() const {
    return path_to_weights_;
}

std::vector<CompilationParameter> compilation_parameters_kmb =
{
    CompilationParameter{"squeezenetv1_1_int8_onnx_0001",
                         "/KMB_models/INT8/squeezenetv1.1-int8-onnx-0001/squeezenetv1.1-int8.xml",
                         "/KMB_models/INT8/squeezenetv1.1-int8-onnx-0001/squeezenetv1.1-int8.bin"},

#if 0
// TODO: SSD512 network can not be parsed into mcmCompiler OpModel due to unsupported layers.
// Jira:
//  Feature VPUNND-1468
//     Normalize layer support
//  Feature VPUNND-1467
//     PriorBox layer support
//  Feature VPUNND-1466
//     Gather layer support
//  Feature VPUNND-1465
//     DetectionOutput layer support
//  Feature VPUNND-1464
//     Unsqueeze layer support
//  Feature VPUNND-1463
//     Squeeze layer support
    CompilationParameter{"SSD512_int8_onnx_0001",
                         "/KMB_models/INT8/SSD512-int8-onnx-0001/SSD512-int8-onnx-0001.xml",
                         "/KMB_models/INT8/SSD512-int8-onnx-0001/SSD512-int8-onnx-0001.bin"},
#endif

    CompilationParameter{"resnet_50_int8_tf_0001",
                         "/KMB_models/INT8/resnet-50-int8-tf-0001/resnet50-int8.xml",
                         "/KMB_models/INT8/resnet-50-int8-tf-0001/resnet50-int8.bin"},
    CompilationParameter{"mobilenetv2_int8_tf_0001",
                         "/KMB_models/INT8/mobilenetv2-int8-tf-0001/mobilenetv2-int8.xml",
                         "/KMB_models/INT8/mobilenetv2-int8-tf-0001/mobilenetv2-int8.bin"},
    CompilationParameter{"inceptionv3_int8_tf_0001",
                         "/KMB_models/INT8/inceptionv3-int8-tf-0001/inceptionv3-int8.xml",
                         "/KMB_models/INT8/inceptionv3-int8-tf-0001/inceptionv3-int8.bin"},
    CompilationParameter{"inceptionv1_int8_tf_0001",
                         "/KMB_models/INT8/inceptionv1-int8-tf-0001/inceptionv1-int8-tf-0001.xml",
                         "/KMB_models/INT8/inceptionv1-int8-tf-0001/inceptionv1-int8-tf-0001.bin"},

    CompilationParameter{"resnet_v1_50_75.19_fp16",
                         "/KMB_models/FP16/resnet_v1_50_75.19/resnet50_v1_fp16.xml",
                         "/KMB_models/FP16/resnet_v1_50_75.19/resnet50_v1_fp16.bin"},
    CompilationParameter{"mobilenet_v2_1.0_224_frozen_71.74_fp16",
                         "/KMB_models/FP16/mobilenet_v2_1.0_224_frozen_71.74/mobilenet_v2_1_no_preprocess.xml",
                         "/KMB_models/FP16/mobilenet_v2_1.0_224_frozen_71.74/mobilenet_v2_1_no_preprocess.bin"},
    CompilationParameter{"inception_v3_74.19_fp16",
                         "/KMB_models/FP16/inception_v3_74.19/inception_v3_no_preprocess.xml",
                         "/KMB_models/FP16/inception_v3_74.19/inception_v3_no_preprocess.bin"},
};

}

inline void VpuNoRegressionWithCompilation::loadNetworkWrapper(std::map<std::string, std::string> config, InferenceEngine::StatusCode* st) {
    StatusCode sts;
    InferenceEngine::ResponseDesc response;
    HeteroPluginPtr plugin(make_plugin_name(pluginName));
    CNNNetReader reader;
    reader.ReadNetwork((ModelsPath() + path_to_files.pathToNetwork()).c_str());
    reader.ReadWeights((ModelsPath() + path_to_files.pathToWeights()).c_str());
    CNNNetwork network = reader.getNetwork();

    ExecutableNetwork exeNetwork;

    // Try to get statistics
    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)network).getStats(&pstats, nullptr);

    if (s == StatusCode::OK && pstats && !pstats->isEmpty()) {
        details::CNNNetworkImplPtr clonedNetwork;
        CNNNetworkInt8Normalizer cnnorm;

        clonedNetwork = cloneNet(network);
        cnnorm.NormalizeNetwork(*clonedNetwork, *pstats);
        sts = plugin->LoadNetwork(exeNetwork, *clonedNetwork, config, &response);
    } else {
        sts = plugin->LoadNetwork(exeNetwork, network, config, &response);
    }

    if (st) {
        *st = sts;
        EXPECT_EQ(StatusCode::OK, sts) << response.msg;
    } else {
        ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    }
}

#ifdef ENABLE_MCM_COMPILER
TEST_P(KmbNoRegressionCompilationOnly, IE2MCM) {
    auto toCompile = get<2>(TestParam::GetParam());
    double tm      = get<3>(TestParam::GetParam());
    std::map<std::string, std::string> config(_config);
    if (toCompile) {
        config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);
        config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)]  = CONFIG_VALUE(YES);
        config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)]  = CONFIG_VALUE(NO);
        const ::testing::TestInfo* const test_info =
                ::testing::UnitTest::GetInstance()->current_test_info();
        config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS_PATH)] = test_info->test_case_name();
        config[VPU_KMB_CONFIG_KEY(MCM_COMPILATION_RESULTS)] = test_info->name();
    } else {
        config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);
    }

    std::string statusMessage;
    int runStatus = runWithTimeout (
            [&](int& status) {
                InferenceEngine::StatusCode st = InferenceEngine::StatusCode::GENERAL_ERROR;
                loadNetworkWrapper(config, &st);
                status = st;
            },
            statusMessage, tm);

    ASSERT_EQ(RunStatus::OK, runStatus) << statusMessage;
}

INSTANTIATE_TEST_CASE_P(
        KmbParsingOnlyTest_smoke_nightly,
        KmbNoRegressionCompilationOnly,
        Combine(Values("kmbPlugin"),
                ValuesIn(compilation_parameters_kmb),
                Values<Compile>(false),
                Values<Timeout>(60.)),
                KmbNoRegressionCompilationOnly::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
        DISABLED_KmbCompilationTest_smoke_nightly,
        // TODO: mcmCompiler can not compile top 6 KeemBay target networks. Jira: VPUNND-1412, VPUNND-1415, VPUNND-1416, VPUNND-1417, VPUNND-1418
        KmbNoRegressionCompilationOnly,
        Combine(Values("kmbPlugin"),
                ValuesIn(compilation_parameters_kmb),
                Values<Compile>(true),
                Values<Timeout>(600.)),
                KmbNoRegressionCompilationOnly::getTestCaseName);
#endif


#ifdef ENABLE_VPUAL
class VpuNoRegressionInference : public Regression::RegressionTests {
public:
    std::string getPluginName() const override {
        return pluginName;
    }

    std::string getDeviceName() const override {
        return "";
    }
protected:
    std::string pluginName = "kmbPlugin";
};

TEST_F(VpuNoRegressionInference, DISABLED_canDoInferenceOnImportedBlob) {  // To be run in manual mode when device is available
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/TwoFramesConvolution/conv.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB", {}));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
    ASSERT_NO_THROW(inferRequest.Infer());
}

using VpuInferAndCompareTests = vpuLayersTests;

TEST_F(VpuInferAndCompareTests, DISABLED_compareInferenceOutputWithReference) {  // To be run in manual mode when device is available
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/SingleConvolutionFP16/SingleConv.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB", {}));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    ConstInputsDataMap inputInfo;
    inputInfo = importedNetwork.GetInputsInfo();

    std::string inputFilePath = ModelsPath() + "/KMB_models/BLOBS/SingleConvolutionFP16/input.bin";
    for (auto & item : inputInfo) {
        Blob::Ptr inputBlob = inferRequest.GetBlob(item.first.c_str());
        ASSERT_TRUE(fromBinaryFile(inputFilePath, inputBlob));
    }

    ASSERT_NO_THROW(inferRequest.Infer());

    ConstOutputsDataMap outputInfo;
    outputInfo = importedNetwork.GetOutputsInfo();

    for (auto & item : outputInfo) {
        Blob::Ptr outputBlob = inferRequest.GetBlob(item.first.c_str());

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
        Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
            outputBlobTensorDesc.getPrecision(),
            outputBlobTensorDesc.getDims(),
            outputBlobTensorDesc.getLayout()));
        referenceOutputBlob->allocate();

        std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/SingleConvolutionFP16/output.bin";
        ASSERT_TRUE(fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

        Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
        Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
        Compare(refFP32, outputFP32, 0.0f);
    }
}

TEST_F(VpuInferAndCompareTests, DISABLED_inferenceWithPreprocessing) {  // To be run in manual mode when device is available
    std::string irXmlPath = ModelsPath() + "/KMB_models/BLOBS/SingleConvolutionFP16/SingleConvolutionFP16.xml";
    std::string weightsPath = ModelsPath() + "/KMB_models/BLOBS/SingleConvolutionFP16/weights.bin";
    CNNNetReader netReader;
    netReader.ReadNetwork(irXmlPath);
    netReader.ReadWeights(weightsPath);

    CNNNetwork network = netReader.getNetwork();
    InputsDataMap inputInfo = network.getInputsInfo();
    for (auto & item : inputInfo) {
        item.second->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    }

    Core ie;
    std::string input_name = inputInfo.begin()->first;
    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = ie.ImportNetwork(input_name, "KMB", {}));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    std::string inputFilePath = ModelsPath() + "/KMB_models/BLOBS/SingleConvolutionFP16/input-300x300.bin";
    for (auto & item : inputInfo) {
        Blob::Ptr inputBlob;
        inputBlob = inferRequest.GetBlob(item.first.c_str());
        ASSERT_TRUE(fromBinaryFile(inputFilePath, inputBlob));
    }

    ASSERT_NO_THROW(inferRequest.Infer());

    ConstOutputsDataMap outputInfo;
    outputInfo = exeNetwork.GetOutputsInfo();

    for (auto & item : outputInfo) {
        Blob::Ptr outputBlob;
        outputBlob = inferRequest.GetBlob(item.first.c_str());

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
        Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
            outputBlobTensorDesc.getPrecision(),
            outputBlobTensorDesc.getDims(),
            outputBlobTensorDesc.getLayout()));
        referenceOutputBlob->allocate();

        std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/SingleConvolutionFP16/output.bin";
        ASSERT_TRUE(fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

        Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
        Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
        Compare(refFP32, outputFP32, 0.0f);
    }
}

struct modelBlobsPaths {
    std::string _graphPath, _inputPath, _outputPath;
};

class VpuInferAndCompareTestsWithParam: public vpuLayersTests,
                             public testing::WithParamInterface< std::tuple<bool, modelBlobsPaths> > {
};

TEST_P(VpuInferAndCompareTestsWithParam, DISABLED_multipleInferRequests) {
    std::tuple<bool, modelBlobsPaths> paramTuple = GetParam();
    bool isSync = std::get<0>(paramTuple);
    modelBlobsPaths blobsPaths = std::get<1>(paramTuple);
    std::string graphSuffix = blobsPaths._graphPath;
    std::string inputSuffix = blobsPaths._inputPath;
    std::string outputSuffix = blobsPaths._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB", {}));

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

    for (auto & item : inputInfo) {
        for(auto currentRequest : requestList) {
            Blob::Ptr inputBlob;
            inputBlob = currentRequest.GetBlob(item.first.c_str());
            ASSERT_TRUE(fromBinaryFile(inputFilePath, inputBlob));
        }
    }

    const int MAX_WAIT = 60000;
    auto requestRoutine = [MAX_WAIT](InferenceEngine::InferRequest request)->void {
        ResponseDesc response;
        ASSERT_NO_THROW(request.StartAsync());
        ASSERT_EQ(StatusCode::OK, request.Wait(MAX_WAIT)) << response.msg;
    };

    if(isSync) {
        // synchronous execution
        for (InferenceEngine::InferRequest & currentRequest : requestList) {
            ASSERT_NO_THROW(currentRequest.Infer());
        }
    } else {
        // asynchronous execution
        std::list<std::thread> requestThreadList;
        for (InferenceEngine::InferRequest & currentRequest : requestList) {
            requestThreadList.push_back(std::thread(requestRoutine, currentRequest));
        }

        for (std::thread & requestThread : requestThreadList) {
            requestThread.join();
        }
    }

    ConstOutputsDataMap outputInfo;
    outputInfo = importedNetwork.GetOutputsInfo();

    std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
    for (auto & item : outputInfo) {
        for (InferenceEngine::InferRequest & inferRequest : requestList) {
            Blob::Ptr outputBlob;
            outputBlob = inferRequest.GetBlob(item.first.c_str());

            TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
            Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
                outputBlobTensorDesc.getPrecision(),
                outputBlobTensorDesc.getDims(),
                outputBlobTensorDesc.getLayout()));
            referenceOutputBlob->allocate();

            ASSERT_TRUE(fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

            Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
            Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
            Compare(refFP32, outputFP32, 0.0f);
        }
    }
}

class VpuAsyncInferWithParam: public vpuLayersTests,
                             public testing::WithParamInterface< modelBlobsPaths > {
};

TEST_P(VpuAsyncInferWithParam, DISABLED_asyncInferCallback) {
    modelBlobsPaths blobsPaths = GetParam();
    std::string graphSuffix = blobsPaths._graphPath;
    std::string inputSuffix = blobsPaths._inputPath;
    std::string outputSuffix = blobsPaths._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB", {}));

    std::list<InferenceEngine::InferRequest> requestList;
    const int REQUEST_LIMIT = 10;
    for (int requestCount = 0; requestCount < REQUEST_LIMIT; requestCount++) {
        InferenceEngine::InferRequest inferRequest;
        ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());
        requestList.push_back(inferRequest);
    }

    std::string inputFilePath = ModelsPath() + inputSuffix;

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    for (auto & item : inputInfo) {
        for(auto currentRequest : requestList) {
            Blob::Ptr inputBlob;
            ASSERT_NO_THROW(inputBlob = currentRequest.GetBlob(item.first.c_str()));
            ASSERT_TRUE(fromBinaryFile(inputFilePath, inputBlob));
        }
    }

    std::mutex requestCounterGuard;
    volatile int completedRequests = 0;
    auto onComplete = [&completedRequests, &requestCounterGuard](void)->void {
        std::lock_guard<std::mutex> incrementLock(requestCounterGuard);
        completedRequests++;
    };

    // asynchronous execution
    for (InferenceEngine::InferRequest & currentRequest : requestList) {
        currentRequest.SetCompletionCallback(onComplete);
        ASSERT_NO_THROW(currentRequest.StartAsync());
    }

    const int MAX_WAIT = 60000;
    auto waitRoutine = [&completedRequests, MAX_WAIT, REQUEST_LIMIT](void)->void {
        std::chrono::system_clock::time_point endTime =
            std::chrono::system_clock::now() +
            std::chrono::milliseconds(MAX_WAIT);

        while (completedRequests < REQUEST_LIMIT) {
            ASSERT_LE(std::chrono::system_clock::now(), endTime);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    };

    std::thread waitThread(waitRoutine);
    waitThread.join();

    ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();

    std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
    for (auto & item : outputInfo) {
        for (InferenceEngine::InferRequest & inferRequest : requestList) {
            Blob::Ptr outputBlob;
            ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(item.first.c_str()));

            TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
            Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
                outputBlobTensorDesc.getPrecision(),
                outputBlobTensorDesc.getDims(),
                outputBlobTensorDesc.getLayout()));
            referenceOutputBlob->allocate();

            ASSERT_TRUE(fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

            Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
            Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
            Compare(refFP32, outputFP32, 0.0f);
        }
    }
}

TEST_P(VpuAsyncInferWithParam, DISABLED_asyncInferCallbackRecursive) {
    modelBlobsPaths blobsPaths = GetParam();
    std::string graphSuffix = blobsPaths._graphPath;
    std::string inputSuffix = blobsPaths._inputPath;
    std::string outputSuffix = blobsPaths._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB", {}));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    std::string inputFilePath = ModelsPath() + inputSuffix;

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    for (auto & item : inputInfo) {
        Blob::Ptr inputBlob;
        ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(item.first.c_str()));
        ASSERT_TRUE(fromBinaryFile(inputFilePath, inputBlob));
    }

    const std::size_t MAX_ITERATIONS = 10;
    volatile std::size_t iterationCount = 0;
    std::condition_variable waitCompletion;

    auto onComplete = [&waitCompletion, &iterationCount, &inferRequest](void)->void {
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
    waitCompletion.wait(execLocker, [&]{ return iterationCount == MAX_ITERATIONS; });

    ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();

    std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
    for (auto & item : outputInfo) {
        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(item.first.c_str()));

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();
        Blob::Ptr referenceOutputBlob = make_blob_with_precision(TensorDesc(
            outputBlobTensorDesc.getPrecision(),
            outputBlobTensorDesc.getDims(),
            outputBlobTensorDesc.getLayout()));
        referenceOutputBlob->allocate();

        ASSERT_TRUE(fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

        Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
        Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
        Compare(refFP32, outputFP32, 0.0f);
    }
}

const static std::vector<bool> isSyncVec = {
    false, true
};

const static std::vector<modelBlobsPaths> pathToPreCompiledGraph = {
    {
        ._graphPath = "/KMB_models/BLOBS/mobilenet/mobilenet.blob",
        ._inputPath = "/KMB_models/BLOBS/mobilenet/input.dat",
        ._outputPath = "/KMB_models/BLOBS/mobilenet/output.dat"
    },
    {
        ._graphPath = "/KMB_models/BLOBS/resnet/resnet.blob",
        ._inputPath = "/KMB_models/BLOBS/resnet/input.dat",
        ._outputPath = "/KMB_models/BLOBS/resnet/output.dat"
    },
    {
        ._graphPath = "/KMB_models/BLOBS/yolotiny/yolotiny.blob",
        ._inputPath = "/KMB_models/BLOBS/yolotiny/input.dat",
        ._outputPath = "/KMB_models/BLOBS/yolotiny/output.dat"
    }
};

INSTANTIATE_TEST_CASE_P(multipleInference, VpuInferAndCompareTestsWithParam,
    ::testing::Combine(
        ::testing::ValuesIn(isSyncVec),
        ::testing::ValuesIn(pathToPreCompiledGraph)
    )
);

INSTANTIATE_TEST_CASE_P(asyncInferenceWithCallback, VpuAsyncInferWithParam,
    ::testing::ValuesIn(pathToPreCompiledGraph)
);

#endif
