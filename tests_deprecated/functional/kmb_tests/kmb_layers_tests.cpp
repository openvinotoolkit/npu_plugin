//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "kmb_layers_tests.hpp"

#include "vpux/utils/core/helper_macros.hpp"

#include <chrono>
#include <iostream>
#include <thread>

#include <vpux/vpux_plugin_config.hpp>
#include <vpux/vpux_compiler_config.hpp>

#include "functional_test_utils/plugin_cache.hpp"

using namespace InferenceEngine;

void kmbLayersTests_nightly::NetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params,
    int weights_size, int biases_size, InferenceEngine::TBlob<uint8_t>::Ptr weights,
    InferenceEngine::Precision outputPrecision, InferenceEngine::Precision inputPrecision) {
#if defined(__arm__) || defined(__aarch64__)
    VPUX_UNUSED(layer_type);
    VPUX_UNUSED(params);
    VPUX_UNUSED(weights_size);
    VPUX_UNUSED(biases_size);
    VPUX_UNUSED(weights);
    VPUX_UNUSED(outputPrecision);
    VPUX_UNUSED(inputPrecision);
    GTEST_SKIP();
#else
    ASSERT_NO_FATAL_FAILURE(
        doNetworkInit(layer_type, params, weights_size, biases_size, weights, outputPrecision, inputPrecision););
#endif
}

void kmbLayersTests_nightly::setup(const CNNNetwork& network, InferenceEngine::Precision outputPrecision,
    InferenceEngine::Precision inputPrecision, bool) {
#if defined(__arm__) || defined(__aarch64__)
    VPUX_UNUSED(network);
    VPUX_UNUSED(outputPrecision);
    VPUX_UNUSED(inputPrecision);
    GTEST_SKIP();
#else
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
#endif
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
    config = this->config;
    config[VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT)] = CONFIG_VALUE(YES);
    config[VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT)] = CONFIG_VALUE(YES);
}
