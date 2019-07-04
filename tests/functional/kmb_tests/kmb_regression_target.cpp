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

#include <thread>
#include <chrono>
#include <gtest/gtest.h>
#include <regression_tests.hpp>
#include <string>
#include <inference_engine/precision_utils.h>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;

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
using CompilationTestParam = WithParamInterface<std::tuple<std::string, CompilationParameter, Compile>>;

class VpuNoRegressionWithCompilation : public Regression::RegressionTests,
                                       public CompilationTestParam {
public:
    using TestParam = WithParamInterface<std::tuple<std::string, CompilationParameter, Compile>>;

    // Operations
    static std::string getTestCaseName(TestParamInfo <CompilationTestParam::ParamType> param);
    inline void loadNetworkWrapper(std::map<std::string, std::string> config);

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
    CompilationParameter{"squeezenet_v1_1-int8",
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

inline void VpuNoRegressionWithCompilation::loadNetworkWrapper(std::map<std::string, std::string> config) {
    StatusCode sts;
    InferenceEngine::ResponseDesc response;
    HeteroPluginPtr plugin(make_plugin_name(pluginName));
    CNNNetReader reader;
    reader.ReadNetwork((ModelsPath() + path_to_files.pathToNetwork()).c_str());
    reader.ReadWeights((ModelsPath() + path_to_files.pathToWeights()).c_str());
    CNNNetwork network = reader.getNetwork();

    IExecutableNetwork::Ptr exeNetwork;

    // Try to get statistics
    ICNNNetworkStats* pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)network).getStats(&pstats, nullptr);

    if (s == StatusCode::OK && pstats && !pstats->isEmpty()) {
        details::CNNNetworkImplPtr clonedNetwork;
        CNNNetworkInt8Normalizer cnnorm;

        config[VPU_CONFIG_KEY(ALLOW_FP32_MODELS)] = CONFIG_VALUE(YES);
        clonedNetwork = cloneNet(network);
        cnnorm.NormalizeNetwork(*clonedNetwork, *pstats);
        sts = plugin->LoadNetwork(exeNetwork, *clonedNetwork, config, &response);
    } else {
        sts = plugin->LoadNetwork(exeNetwork, network, config, &response);
    }

    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

#ifdef ENABLE_MCM_COMPILER
TEST_P(KmbNoRegressionCompilationOnly, IE2MCM) {
    auto toCompile = get<2>(TestParam::GetParam());
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

    loadNetworkWrapper(config);
}

INSTANTIATE_TEST_CASE_P(
        KmbParsingOnlyTest_smoke_nightly,
        KmbNoRegressionCompilationOnly,
        Combine(Values("kmbPlugin"),
                ValuesIn(compilation_parameters_kmb),
                Values<Compile>(false)),
                KmbNoRegressionCompilationOnly::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
        DISABLED_KmbCompilationTest_smoke_nightly,
        // TODO: mcmCompiler can not compile top 6 KeemBay target networks. Jira: VPUNND-1412, VPUNND-1415, VPUNND-1416, VPUNND-1417, VPUNND-1418
        KmbNoRegressionCompilationOnly,
        Combine(Values("kmbPlugin"),
                ValuesIn(compilation_parameters_kmb),
                Values<Compile>(true)),
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

TEST_F(VpuNoRegressionInference, canDoInferenceOnImportedBlob) {
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/TwoFramesConvolution/conv.blob";

    Core ie;
    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr = ie.ImportNetwork(modelFilePath, modelFilePath, {});
    ASSERT_NE(nullptr, importedNetworkPtr);

    InferenceEngine::IInferRequest::Ptr inferRequest;
    ResponseDesc resp;
    ASSERT_EQ(StatusCode::OK, importedNetworkPtr->CreateInferRequest(inferRequest, &resp)) << resp.msg;

    ASSERT_EQ(StatusCode::OK, inferRequest->Infer(&resp)) << resp.msg;
}
#endif

