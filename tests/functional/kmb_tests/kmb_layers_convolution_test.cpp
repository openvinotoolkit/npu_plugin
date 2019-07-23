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

#include <fstream>
#include <vpu/kmb_plugin_config.hpp>

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;
using namespace InferenceEngine::details;

#ifdef ENABLE_MCM_COMPILER
TEST_F(kmbLayersTests_nightly, DISABLED_TestsConvolutionAfterScaleShift) {
    // TODO: tests fails. mcmCompiler compilation (Convolution with bias): Segmentation fault. Jira: VPUNND-1474
    const std::string model = conv_after_scale_shift;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    _exeNetwork = ie.LoadNetwork(network, "kmb", config);
}

TEST_F(kmbLayersTests_nightly, TestsConvolutionAfterScaleShiftNoBias) {
    std::string model = conv_after_scale_shift;
    REPLACE_WITH_STR(model, "<biases offset=\"6\" size=\"6\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"18828\" size=\"128\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    _exeNetwork = ie.LoadNetwork(network, "kmb", config);
}

TEST_F(kmbLayersTests_nightly, DISABLED_TestsQuantizedConvolutionAfterScaleShift) {
    // TODO: Test fails. mcmCompiler can not compile the network (Convolution with bias). Jira: VPUNND-1474
    const std::string model = full_quant_model;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    StatusCode sts;
    InferenceEngine::ResponseDesc response;
    std::map<std::string, std::string> config;
    IExecutableNetwork::Ptr exeNetwork;
    details::CNNNetworkImplPtr clonedNetwork;

    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    std::size_t weightSize = 147456 + 65536;
    std::size_t biasSize = 256 + 1024;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightSize + biasSize));
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
        InferenceEngine::details::CNNNetworkInt8Normalizer::NormalizeNetwork(*clonedNetwork, *pstats);

        _exeNetwork = ie.LoadNetwork(CNNNetwork(clonedNetwork), "kmb", config);
    }
}

TEST_F(kmbLayersTests_nightly, TestsQuantizedConvolutionAfterScaleShiftNoBias) {
    std::string model = full_quant_model;

    REPLACE_WITH_STR(model, "<biases offset=\"147456\" size=\"256\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"213248\" size=\"1024\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    StatusCode sts;
    InferenceEngine::ResponseDesc response;
    std::map<std::string, std::string> config;
    IExecutableNetwork::Ptr exeNetwork;
    details::CNNNetworkImplPtr clonedNetwork;
    CNNNetworkInt8Normalizer cnnorm;

    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);

    std::size_t weightSize = 147456 + 65536;
    std::size_t biasSize = 256 + 1024;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightSize + biasSize));
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
        InferenceEngine::details::CNNNetworkInt8Normalizer::NormalizeNetwork(*clonedNetwork, *pstats);

        _exeNetwork = ie.LoadNetwork(CNNNetwork(clonedNetwork), "kmb", config);
    }

}

TEST_F(kmbLayersTests_nightly, TestsConvolutionOnly) {
    const std::string model = convolution_only;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 18816;
    std::size_t biasSize = 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    // LoadNetwork results in the following message when MCM_PARSING_ONLY is set to 'NO':
    // The maximum peak memory requirment of the graph exceeds CMX and the partial serialisation algorithm is unable
    // to reduce parallelism, exiting now, this is normal behaviour
    // TODO disable 'parse only' and find out why it happens
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);

    _exeNetwork = ie.LoadNetwork(network, "kmb", config);
}

TEST_F(kmbLayersTests_nightly, TestsConvolutionOnlyNoBias) {
    std::string model = convolution_only;
    REPLACE_WITH_STR(model, "<biases offset=\"18816\" size=\"128\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    std::size_t weightSize = 18816;
    std::size_t biasSize = 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightSize + biasSize));
    ASSERT_NO_THROW(_net_reader.SetWeights(weightsBlob));

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["conv_test1"]->setPrecision(Precision::FP16);

    std::map<std::string, std::string> config;
    // LoadNetwork results in the following message when MCM_PARSING_ONLY is set to 'NO':
    // The maximum peak memory requirment of the graph exceeds CMX and the partial serialisation algorithm is unable
    // to reduce parallelism, exiting now, this is normal behaviour
    // TODO disable 'parse only' and find out why it happens
    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(YES);

    _exeNetwork = ie.LoadNetwork(network, "kmb", config);
}


// This function is necessary to check that two files with blobs have the same size (see test below)
int getFileSize(const std::string &fileName) {
    std::ifstream file(fileName.c_str(), std::ifstream::in | std::ifstream::binary);

    if(!file.is_open()) {
        std::cout << "Can not open file: " << fileName << std::endl;
        return -1;
    }

    file.seekg(0, std::ios::end);
    int sizeOfFile = file.tellg();
    file.close();

    return sizeOfFile;
}

// Function for comparison of content of two blob files
bool isContentOfFilesEqual (const std::string &fileName1, const std::string &fileName2){
    std::ifstream file1(fileName1.c_str(), std::ifstream::in | std::ifstream::binary);
    std::ifstream file2(fileName2.c_str(), std::ifstream::in | std::ifstream::binary);

    if( !file1.is_open() || !file2.is_open() ) {
        std::cout << "Can not open file! " << std::endl;
        return false;
    }

    char x, y;

    while ( !file1.eof() || !file2.eof() )
    {
        file1.read(&x, 1);
        file2.read(&y, 1);
        if ( x != y )
            return false;
    }
    return true;
}


TEST_F(kmbLayersTests_nightly, TestExportImportBlob01) {
    std::string model = full_quant_model;

    REPLACE_WITH_STR(model, "<biases offset=\"147456\" size=\"256\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"213248\" size=\"1024\"/>", " ");

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    StatusCode sts;
    InferenceEngine::ResponseDesc response;
    std::map<std::string, std::string> config;
    IExecutableNetwork::Ptr exeNetwork;
    details::CNNNetworkImplPtr clonedNetwork;
    CNNNetworkInt8Normalizer cnnorm;

    setCommonConfig(config);
    config[VPU_KMB_CONFIG_KEY(MCM_PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_BLOB)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_DOT)] = CONFIG_VALUE(YES);
    config[VPU_KMB_CONFIG_KEY(MCM_GENERATE_JSON)] = CONFIG_VALUE(YES);
    config[VPU_CONFIG_KEY(ALLOW_FP32_MODELS)] = CONFIG_VALUE(YES);

    std::size_t weightSize = 147456 + 65536;
    std::size_t biasSize = 256 + 1024;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights(weightSize + biasSize));
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
        cnnorm.NormalizeNetwork(*clonedNetwork, *pstats);
        sts = myriadPluginPtr->LoadNetwork(_exeNetwork, *clonedNetwork, config, &response);
    }

    ASSERT_EQ(StatusCode::OK, sts) << _resp.msg;

    std::string blobFileName = "TestExportBlob01.blob";
    std::cout << "Try to write blob to file" << std::endl;
    s = _exeNetwork->Export(blobFileName, nullptr);
    ASSERT_EQ(StatusCode::OK, s);

    Core ie;
    std::cout << "Try to load blob from file" << std::endl;
    InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr = ie.ImportNetwork(blobFileName, "KMB", {});
    ASSERT_NE(nullptr, importedNetworkPtr);

    std::string blobFileName2 = "TestExportBlob02.blob";
    std::cout << "Try to write blob to file again" << std::endl;
    s = importedNetworkPtr->Export(blobFileName2, nullptr);
    ASSERT_EQ(StatusCode::OK, s);

    std::cout << "Compare size of base and derivated executable networks" << std::endl;
    ASSERT_EQ(getFileSize(blobFileName), getFileSize(blobFileName2));

    std::cout << "Compare content of base and derivated executable networks" << std::endl;
    ASSERT_TRUE( isContentOfFilesEqual(blobFileName, blobFileName2) );

}
#endif
