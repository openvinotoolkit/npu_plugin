// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

#include <thread>
#include <chrono>
#include <iostream>

#include "plugin_cache.hpp"

using namespace InferenceEngine;

void myriadLayersTests_nightly::NetworkInit(const std::string& layer_type,
                std::map<std::string, std::string>* params,
                int weights_size,
                int biases_size,
                InferenceEngine::TBlob<uint8_t>::Ptr weights,
                InferenceEngine::Precision outputPrecision,
                InferenceEngine::Precision inputPrecision,
                bool useHWOpt)
{
    ASSERT_NO_FATAL_FAILURE(
        doNetworkInit(layer_type,
                      params,
                      weights_size,
                      biases_size,
                      weights,
                      outputPrecision,
                      inputPrecision,
                      useHWOpt);
    );
}

void myriadLayersTests_nightly::doNetworkInit(const std::string& layer_type,
                std::map<std::string, std::string>* params,
                int weights_size,
                int biases_size,
                InferenceEngine::TBlob<uint8_t>::Ptr weights,
                InferenceEngine::Precision outputPrecision,
                InferenceEngine::Precision inputPrecision,
                bool useHWOpt)
{
    std::string xml;
    genXML(layer_type, params, weights_size, biases_size, xml);
    ASSERT_NO_THROW(_net_reader.ReadNetwork(xml.data(), xml.length()));
    ASSERT_EQ(_net_reader.isParseSuccess(), true);
    if (weights != nullptr)
        ASSERT_NO_THROW(_net_reader.SetWeights(weights));
    setup(outputPrecision, inputPrecision, useHWOpt);
}

const std::vector<InferenceEngine::SizeVector> g_poolingInput = {
    {{1,  1,  16,  16},
     {1,  8, 228, 128},
     {1, 16,  32,  64}}
};

const std::vector<InferenceEngine::SizeVector> g_poolingInput_postOp = {
    {{1, 32,  86, 100}, // postOp issue MX
     {1, 32,  62, 104}} // postOp issue M2
};

const std::vector<pooling_layer_params> g_poolingLayerParamsFull = {
    /* kernel stride  pad */
    {{2, 2}, {2, 2}, {0, 0}},
    {{2, 2}, {2, 2}, {1, 1}},
    {{2, 2}, {2, 2}, {2, 2}},
    {{2, 2}, {1, 1}, {0, 0}},
    {{2, 2}, {1, 1}, {1, 1}},
    {{2, 2}, {1, 1}, {2, 2}},
    {{4, 2}, {2, 2}, {0, 0}},
    {{4, 2}, {2, 2}, {1, 1}},
    {{4, 2}, {2, 2}, {2, 2}},
    {{4, 2}, {1, 1}, {0, 0}},
    {{4, 2}, {1, 1}, {1, 1}},
    {{4, 2}, {1, 1}, {2, 2}},
    {{2, 4}, {2, 2}, {0, 0}},
    {{2, 4}, {2, 2}, {1, 1}},
    {{2, 4}, {2, 2}, {2, 2}},
    {{2, 4}, {1, 1}, {0, 0}},
    {{2, 4}, {1, 1}, {1, 1}},
    {{2, 4}, {1, 1}, {2, 2}},
};

const std::vector<pooling_layer_params> g_poolingLayerParamsLite = {
    /* kernel stride  pad */
    {{2, 2}, {1, 1}, {0, 0}},
    {{4, 2}, {2, 2}, {1, 1}},
    {{2, 4}, {1, 1}, {2, 2}},
};

const std::vector<const char*> g_poolingLayout = {
    VPU_CONFIG_VALUE(NCHW),
    VPU_CONFIG_VALUE(NHWC),
};

const std::vector<InferenceEngine::SizeVector> g_convolutionTensors = {
    {{1, 8, 4, 16}, {16, 8, 16}}  //NCHW
};

const std::vector<InferenceEngine::SizeVector> g_convolutionTensors_postOp = {
    {{1, 32, 112, 96}}  /* postOp issue */
};

const std::vector<fcon_test_params> g_fcTestParamsSubset = {
    {{1, 4, 8, 16}, 4, 0.065f},
    {{1, 16, 8, 8}, 8, 0.065f}
};

/* tests subset to check 2 layers operation invocation */
/* additional tests for 2D and 3D tensors added        */
const std::vector<int32_t> g_dimensionsFC = {
    4, 3
};

const std::vector<int32_t> g_addBiasFC = {
    1, 0
};
