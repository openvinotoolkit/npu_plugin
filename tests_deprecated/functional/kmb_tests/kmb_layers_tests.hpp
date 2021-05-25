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

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <ie_version.hpp>
#include <inference_engine.hpp>
#include <tuple>
#include <vpux/vpux_plugin_config.hpp>
#include <vpux/vpux_compiler_config.hpp>

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include "vpu_layers_tests.hpp"

class kmbLayersTests_nightly : public vpuLayersTests {
public:
    void NetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params = nullptr,
        int weights_size = 0, int biases_size = 0, InferenceEngine::TBlob<uint8_t>::Ptr weights = nullptr,
        InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32,
        InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16);

    void setCommonConfig(std::map<std::string, std::string>& config);

private:
    void doNetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params = nullptr,
        int weights_size = 0, int biases_size = 0, InferenceEngine::TBlob<uint8_t>::Ptr weights = nullptr,
        InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32,
        InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16);

    void setup(const CNNNetwork& network, InferenceEngine::Precision outputPrecision,
        InferenceEngine::Precision inputPrecision, bool useHWOpt = false) override;
};

template <class T>
class kmbLayerTestBaseWithParam : public kmbLayersTests_nightly, public testing::WithParamInterface<T> {};

class KmbPerLayerTest : public ::testing::Test {
public:
    std::map<std::string, std::string> getCommonConfig() const;
    std::string getTestResultFilename() const;
};
