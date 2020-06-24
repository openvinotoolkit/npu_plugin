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

#include <chrono>
#include <iostream>
#include <thread>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/vpu_compiler_config.hpp>

#include "functional_test_utils/plugin_cache.hpp"

using namespace InferenceEngine;

void kmbLayersTests_nightly::NetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params,
    int weights_size, int biases_size, InferenceEngine::TBlob<uint8_t>::Ptr weights,
    InferenceEngine::Precision outputPrecision, InferenceEngine::Precision inputPrecision) {
    ASSERT_NO_FATAL_FAILURE(
        doNetworkInit(layer_type, params, weights_size, biases_size, weights, outputPrecision, inputPrecision););
}

void kmbLayersTests_nightly::setup(
    const CNNNetwork& network, InferenceEngine::Precision outputPrecision, InferenceEngine::Precision inputPrecision) {
    _inputsInfo = network.getInputsInfo();
    for (const auto& in : _inputsInfo) {
        in.second->setPrecision(inputPrecision);
        in.second->setLayout(InferenceEngine::Layout::NHWC);
    }
    _outputsInfo = network.getOutputsInfo();
    for (const auto& outputInfo : _outputsInfo) {
        outputInfo.second->setPrecision(outputPrecision);
        outputInfo.second->setLayout(InferenceEngine::Layout::NHWC);
    }

    std::map<std::string, std::string> config;
    setCommonConfig(config);
    ASSERT_NO_THROW(core->LoadNetwork(network, deviceName, config));
}

void kmbLayersTests_nightly::doNetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params,
    int weights_size, int biases_size, InferenceEngine::TBlob<uint8_t>::Ptr weights,
    InferenceEngine::Precision outputPrecision, InferenceEngine::Precision inputPrecision) {
    std::string xml;
    genXML(layer_type, params, weights_size, biases_size, xml);
    CNNNetwork network;
    ASSERT_NO_THROW(network = core->ReadNetwork(xml, weights));
    setup(network, outputPrecision, inputPrecision);
}

std::map<std::string, std::string> KmbPerLayerTest::getCommonConfig() const {
    std::map<std::string, std::string> config;
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);

    return config;
}

std::string KmbPerLayerTest::getTestResultFilename() const {
    std::string testResultFilename = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    for (auto& letter : testResultFilename) {
        letter = (letter == '/') ? '_' : letter;
    }

    return testResultFilename;
}

void kmbLayersTests_nightly::setCommonConfig(std::map<std::string, std::string>& config) {
    config = config;
    config[VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_JSON)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(GENERATE_DOT)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(PARSING_ONLY)] = CONFIG_VALUE(NO);
    config[VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT)] = CONFIG_VALUE(YES);

    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();

    config[VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS_PATH)] = test_info->test_case_name();
    config[VPU_COMPILER_CONFIG_KEY(COMPILATION_RESULTS)] = test_info->name();
}
