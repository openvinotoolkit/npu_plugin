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
#include "kmb_layers_tests.hpp"
#include "kmb_regression_target.hpp"
#include "kmb_xml_tests.hpp"

#include <gtest/gtest.h>
#include <regression_tests.hpp>
#include <inference_engine/precision_utils.h>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>
#include "low_precision_transformations/transformer.hpp"

#include <vpu_layers_tests.hpp>

#include <mutex>
#include <condition_variable>
#include <ie_layers.h>

#include <file_reader.h>
#include <blob_factory.hpp>

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;
using namespace TestsTimeout;
using namespace KmbRegressionTarget;

namespace
{

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
        std::string getDeviceName() const override ;
        std::map<std::string, std::string> _config;

    protected:
        // Data section
        std::string pluginName;
        CompilationParameter path_to_files;

        //Operations
        void SetUp() override;
    };

#ifdef ENABLE_MCM_COMPILER
    std::string VpuNoRegressionWithCompilation::getTestCaseName(
            TestParamInfo <CompilationTestParam::ParamType> param) {
        return std::string("Main_") +
               get<0>(param.param) +
               std::string("_") + get<1>(param.param).name +
               ((get<2>(param.param)) ? std::string("_Compilation") : std::string("_Parsing"));
    }

    void VpuNoRegressionWithCompilation::SetUp() {
        pluginName = get<0>(TestParam::GetParam());
        path_to_files = get<1>(TestParam::GetParam());

        PluginCache::get().reset();
    }
#endif

    inline std::string VpuNoRegressionWithCompilation::getDeviceName() const {
        return "";
    }

    class KmbNoRegressionCompilationOnly : public VpuNoRegressionWithCompilation {
    };

    std::vector<CompilationParameter> compilation_parameters_unsupported = {
// TODO: Yolo network can't be parsed into mcmCompiler due to bug in regionYolo parsing
            CompilationParameter{"darknet_tiny_yolo_voc",
                                 "/KMB_models/INT8/darknet_tiny_yolo_voc/yolov2_IR_i8.xml",
                                 "/KMB_models/INT8/darknet_tiny_yolo_voc/yolov2_IR_i8.bin"},
            CompilationParameter{"darknet19_yolo_voc",
                                 "/KMB_models/INT8/darknet19_yolo_voc/yolov2_IR_i8.xml",
                                 "/KMB_models/INT8/darknet19_yolo_voc/yolov2_IR_i8.bin"},
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
    };

    std::vector<CompilationParameter> compilation_parameters_kmb = {
            CompilationParameter{"squeezenetv1_1_int8_onnx_0001",
                                 "/KMB_models/INT8/squeezenetv1.1-int8-onnx-0001/squeezenetv1.1-int8.xml",
                                 "/KMB_models/INT8/squeezenetv1.1-int8-onnx-0001/squeezenetv1.1-int8.bin"},
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

}  // namespace

inline void VpuNoRegressionWithCompilation::loadNetworkWrapper(std::map<std::string, std::string> config, InferenceEngine::StatusCode* st) {
    StatusCode sts;
    InferenceEngine::ResponseDesc response;
    InferenceEnginePluginPtr plugin(make_plugin_name(pluginName));
    CNNNetReader reader;
    reader.ReadNetwork((ModelsPath() + path_to_files.path_to_network).c_str());
    reader.ReadWeights((ModelsPath() + path_to_files.path_to_weights).c_str());
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
TEST_P(KmbNoRegressionCompilationOnly, DISABLED_IE2MCM) {
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

INSTANTIATE_TEST_CASE_P(
        DISABLED_KmbParsingUnsupportedOnlyTest_smoke_nightly,
        KmbNoRegressionCompilationOnly,
        Combine(Values("kmbPlugin"),
                ValuesIn(compilation_parameters_unsupported),
                Values<Compile>(false),
                Values<Timeout>(60.)),
        KmbNoRegressionCompilationOnly::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
        DISABLED_KmbCompilationParsingUnsupportedTest_smoke_nightly,
        KmbNoRegressionCompilationOnly,
        Combine(Values("kmbPlugin"),
                ValuesIn(compilation_parameters_unsupported),
                Values<Compile>(true),
                Values<Timeout>(600.)),
        KmbNoRegressionCompilationOnly::getTestCaseName);

using kmbLayersTestsConvolution = kmbLayersTests_nightly;

TEST_F(kmbLayersTestsConvolution, compilationLoadNetworkAndInfer) {
    std::string model = convolution_u8_only;

    const size_t convolutionWeightsByteSize = 36864;
    const size_t convolutionWeightsSize = convolutionWeightsByteSize / sizeof(uint8_t);

    const size_t biasByteSize = 256;
    const size_t biasSize = biasByteSize / sizeof(int32_t);

    auto convolutionWeightsBuffer = make_shared_blob<uint8_t>({Precision::U8, {convolutionWeightsByteSize + biasByteSize}, Layout::C});
    convolutionWeightsBuffer->allocate();
    auto weightsBufferData = convolutionWeightsBuffer->buffer().as<uint8_t*>();
    for (size_t i = 0; i < convolutionWeightsSize; ++i) {
        weightsBufferData[i] = 1;
    }

    uint32_t* biasData = reinterpret_cast<uint32_t*>(convolutionWeightsBuffer->buffer().as<uint8_t*>() + convolutionWeightsSize);
    for (size_t i = 0; i < biasSize; ++i) {
        biasData[i] = 1lu;
    }

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_NO_THROW(_net_reader.SetWeights(convolutionWeightsBuffer));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    auto network = _net_reader.getNetwork();

    auto _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::U8);

    auto _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv2"]->setPrecision(Precision::U8);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);

    Core ie;
    InferenceEngine::ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(network, "KMB", config));

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

#ifdef ENABLE_VPUAL

const size_t NUMBER_OF_TOP_CLASSES = 5;
const std::string YOLO_GRAPH_NAME = "yolotiny.blob";

struct modelBlobsInfo {
    std::string _graphPath, _inputPath, _outputPath;
};

const static std::vector<modelBlobsInfo> pathToPreCompiledGraph = {
        {
                ._graphPath = "/KMB_models/BLOBS/mobilenet/mobilenet.blob",
                ._inputPath = "/KMB_models/BLOBS/mobilenet/input.dat",
                ._outputPath = "/KMB_models/BLOBS/mobilenet/output.dat",
        },
        {
                ._graphPath = "/KMB_models/BLOBS/resnet/resnet.blob",
                ._inputPath = "/KMB_models/BLOBS/resnet/input.dat",
                ._outputPath = "/KMB_models/BLOBS/resnet/output.dat",
        },
        {
                ._graphPath = "/KMB_models/BLOBS/yolotiny/yolotiny.blob",
                ._inputPath = "/KMB_models/BLOBS/yolotiny/input.dat",
                ._outputPath = "/KMB_models/BLOBS/yolotiny/output.dat",
        }
};


class VpuInferWithPath: public vpuLayersTests,
                        public testing::WithParamInterface< modelBlobsInfo > {
};

class VpuNoRegressionInference : public Regression::RegressionTests {
public:
    std::string getDeviceName() const override {
        return "";
    }
protected:
    std::string pluginName = "kmbPlugin";
};

TEST_P(VpuInferWithPath, canDoInferenceOnImportedBlob) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string modelFilePath = ModelsPath() + blobsInfo._graphPath;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

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
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    ConstInputsDataMap inputInfo;
    inputInfo = importedNetwork.GetInputsInfo();

    std::string inputFilePath = ModelsPath() + inputSuffix;
    for (auto & item : inputInfo) {
        Blob::Ptr inputBlob = inferRequest.GetBlob(item.first.c_str());
        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
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

        std::string referenceOutputFilePath = ModelsPath() + outputSuffix;
        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));
        if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
            Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
            Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
            Compare(refFP32, outputFP32, 0.0f);
        } else {
            ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_TOP_CLASSES));
        }
    }
}

class VpuInferAndCompareTestsWithParam: public vpuLayersTests,
                             public testing::WithParamInterface< std::tuple<bool, modelBlobsInfo> > {
};

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
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

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
            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
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

            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));
            if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
                Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
                Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
                Compare(refFP32, outputFP32, 0.0f);
            } else {
                ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_TOP_CLASSES));
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

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

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
            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
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

            ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));
            if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
                Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
                Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
                Compare(refFP32, outputFP32, 0.0f);
            } else {
                ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_TOP_CLASSES));
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

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    std::string inputFilePath = ModelsPath() + inputSuffix;

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    for (auto & item : inputInfo) {
        Blob::Ptr inputBlob;
        ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(item.first.c_str()));
        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, inputBlob));
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

        ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

        if (graphSuffix.rfind(YOLO_GRAPH_NAME) == graphSuffix.size() - YOLO_GRAPH_NAME.size()) {
            Blob::Ptr refFP32 = ConvertU8ToFP32(referenceOutputBlob);
            Blob::Ptr outputFP32 = ConvertU8ToFP32(outputBlob);
            Compare(refFP32, outputFP32, 0.0f);
        } else {
            ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_TOP_CLASSES));
        }
    }
}

const static std::vector<bool> isSyncVec = {
    false, true
};

INSTANTIATE_TEST_CASE_P(multipleInference, VpuInferAndCompareTestsWithParam,
    ::testing::Combine(
        ::testing::ValuesIn(isSyncVec),
        ::testing::ValuesIn(pathToPreCompiledGraph)
    )
);

TEST_P(VpuInferWithPath, compareSetBlobAndGetBlob) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();
    std::string input_name = inputInfo.begin()->first;

    InferenceEngine::TensorDesc inputTensorDesc = inferRequest.GetBlob(input_name)->getTensorDesc();

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, inputTensorDesc.getDims(), Layout::NCHW}));
    inputBlob->allocate();

    u_int8_t* ptr = inputBlob->buffer();
    memset(ptr, 0xFF, inputBlob->byteSize());
    ASSERT_NO_THROW(inferRequest.SetBlob(input_name, inputBlob));
    Blob::Ptr newInputBlob;
    ASSERT_NO_THROW(newInputBlob = inferRequest.GetBlob(input_name));

    ASSERT_EQ(StatusCode::OK, memcmp(inputBlob->buffer(), newInputBlob->buffer(), inputBlob->byteSize()));
    ASSERT_EQ((void *)inputBlob->buffer(), (void *)newInputBlob->buffer());
}

TEST_P(VpuInferWithPath, compareSetBlobAndGetBlobAfterInfer) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;
    std::string inputNameFilePath = ModelsPath() + inputSuffix;
    std::string outputNameFilePath = ModelsPath() + outputSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

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

    Blob::Ptr fileOutputBlob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, outputTensorDesc.getDims(), Layout::NCHW});
    fileOutputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, fileOutputBlob));


    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, inputTensorDesc.getDims(), Layout::NCHW}));
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
        Blob::Ptr blobFP32 = ConvertU8ToFP32(outputBlob2);
        Blob::Ptr expectedBlobFP32 = ConvertU8ToFP32(outputBlob1);
        Compare(blobFP32, expectedBlobFP32, 0.0f);

        blobFP32 = ConvertU8ToFP32(outputBlob2);
        expectedBlobFP32 = ConvertU8ToFP32(fileOutputBlob);
        Compare(blobFP32, expectedBlobFP32, 0.0f);
    } else {
        ASSERT_NO_THROW(compareTopClasses(outputBlob2, outputBlob1, NUMBER_OF_TOP_CLASSES));
        ASSERT_NO_THROW(compareTopClasses(outputBlob2, fileOutputBlob, NUMBER_OF_TOP_CLASSES));
    }
}

using kmbSetBlob = vpuLayersTests;

TEST_F(kmbSetBlob, compareSetBlobAllocation) {
    std::string mobilenetModelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet/mobilenet.blob";
    std::string resnetModelFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet/resnet.blob";
    std::string inputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet/input.dat";

    Core ie;
    InferenceEngine::ExecutableNetwork mobilNImportedNetwork;
    ASSERT_NO_THROW(mobilNImportedNetwork = ie.ImportNetwork(mobilenetModelFilePath, "KMB"));
    InferenceEngine::ExecutableNetwork resNImportedNetwork;
    ASSERT_NO_THROW(resNImportedNetwork = ie.ImportNetwork(resnetModelFilePath, "KMB"));


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
    ASSERT_EQ((void *)mobilNInputBlob->buffer(), (void *)resNInputBlob->buffer());
}

TEST_P(VpuInferWithPath, compareOutputsTwoNetworks) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;
    std::string inputNameFilePath = ModelsPath() + inputSuffix;
    std::string outputNameFilePath = ModelsPath() + outputSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork1;
    ASSERT_NO_THROW(importedNetwork1 = ie.ImportNetwork(modelFilePath, "KMB"));
    InferenceEngine::ExecutableNetwork importedNetwork2;
    ASSERT_NO_THROW(importedNetwork2 = ie.ImportNetwork(modelFilePath, "KMB"));


    InferenceEngine::InferRequest inferRequest1;
    ASSERT_NO_THROW(inferRequest1 = importedNetwork1.CreateInferRequest());

    std::string input_name1 = importedNetwork1.GetInputsInfo().begin()->first;
    std::string output_name1 = importedNetwork1.GetOutputsInfo().begin()->first;

    InferenceEngine::TensorDesc outputTensorDesc = inferRequest1.GetBlob(output_name1)->getTensorDesc();

    Blob::Ptr fileOutputBlob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, outputTensorDesc.getDims(), Layout::NCHW});
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
        Blob::Ptr blobFP32 = ConvertU8ToFP32(outputBlob2);
        Blob::Ptr expectedBlobFP32 = ConvertU8ToFP32(outputBlob1);
        Compare(blobFP32, expectedBlobFP32, 0.0f);

        blobFP32 = ConvertU8ToFP32(outputBlob2);
        expectedBlobFP32 = ConvertU8ToFP32(fileOutputBlob);
        Compare(blobFP32, expectedBlobFP32, 0.0f);
    } else {
        ASSERT_NO_THROW(compareTopClasses(outputBlob2, outputBlob1, NUMBER_OF_TOP_CLASSES));
        ASSERT_NO_THROW(compareTopClasses(outputBlob2, fileOutputBlob, NUMBER_OF_TOP_CLASSES));
    }
}

TEST_P(VpuInferWithPath, compareSetBlobAndInfer) {
    modelBlobsInfo blobsInfo = GetParam();
    std::string graphSuffix = blobsInfo._graphPath;
    std::string inputSuffix = blobsInfo._inputPath;
    std::string outputSuffix = blobsInfo._outputPath;
    std::string modelFilePath = ModelsPath() + graphSuffix;
    std::string inputNameFilePath = ModelsPath() + inputSuffix;
    std::string outputNameFilePath = ModelsPath() + outputSuffix;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    std::string input_name = importedNetwork.GetInputsInfo().begin()->first;
    std::string output_name = importedNetwork.GetOutputsInfo().begin()->first;

    InferenceEngine::TensorDesc inputTensorDesc = inferRequest.GetBlob(input_name)->getTensorDesc();
    InferenceEngine::TensorDesc outputTensorDesc = inferRequest.GetBlob(output_name)->getTensorDesc();

    Blob::Ptr fileOutputBlob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, outputTensorDesc.getDims(), Layout::NCHW});
    fileOutputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, fileOutputBlob));

    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(inputBlob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, inputTensorDesc.getDims(), Layout::NCHW}));
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
    virtual void SetUp() {
        ptr = std::cout.rdbuf();
        std::cout.rdbuf(out.rdbuf());
    }

    virtual void TearDown() {
        std::cout.rdbuf(ptr);
    }
};

TEST_F(vpuInferWithSetUp, copyCheckSetBlob) {
    std::string strToCheck = "isValidPtr(): Input blob will be copied";
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet/mobilenet.blob";
    std::string inputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet/input.dat";
    std::string outputNameFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet/output.dat";

    Blob::Ptr fileOutputBlob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, {1, 1, 1, 1024}, Layout::NCHW});
    fileOutputBlob->allocate();
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::fromBinaryFile(outputNameFilePath, fileOutputBlob));

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB"));

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

    Blob::Ptr blobFP32 = ConvertU8ToFP32(outputBlob);
    Blob::Ptr expectedBlobFP32 = ConvertU8ToFP32(fileOutputBlob);
    Compare(blobFP32, expectedBlobFP32, 0.0f);

    ASSERT_TRUE(out.str().find(strToCheck) == std::string::npos);
}

INSTANTIATE_TEST_CASE_P(inferenceWithParameters, VpuInferWithPath,
    ::testing::ValuesIn(pathToPreCompiledGraph)
);

#endif
