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

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <ie_version.hpp>
#include <inference_engine.hpp>
#include <tuple>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/vpu_compiler_config.hpp>

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include "vpu_layers_tests.hpp"

class kmbLayersTests_nightly : public vpuLayersTests {
public:
    void NetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params = nullptr,
        int weights_size = 0, int biases_size = 0, InferenceEngine::TBlob<uint8_t>::Ptr weights = nullptr,
        InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32,
        InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16, bool useHWOpt = false);

    void setCommonConfig(std::map<std::string, std::string>& config);

private:
    void doNetworkInit(const std::string& layer_type, std::map<std::string, std::string>* params = nullptr,
        int weights_size = 0, int biases_size = 0, InferenceEngine::TBlob<uint8_t>::Ptr weights = nullptr,
        InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32,
        InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16);

    void setup(const CNNNetwork& network, InferenceEngine::Precision outputPrecision,
        InferenceEngine::Precision inputPrecision, bool useHWOpt = false);
};

template <class T>
class kmbLayerTestBaseWithParam : public kmbLayersTests_nightly, public testing::WithParamInterface<T> {};

class KmbPerLayerTest : public ::testing::Test {
public:
    std::map<std::string, std::string> getCommonConfig() const;
    std::string getTestResultFilename() const;
};
