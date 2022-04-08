//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "kmb_regression_target.hpp"

#include <file_reader.h>
#include <gtest/gtest.h>
#include <precision_utils.h>

#include <blob_factory.hpp>
#include <condition_variable>
#include <mutex>
#include <vpu_layers_tests.hpp>
#include <vpux/vpux_plugin_config.hpp>

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"
#include "tests_timeout.hpp"
#include "yolo_helpers.hpp"

#include "vpux/utils/IE/blob.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace TestsTimeout;
using namespace KmbRegressionTarget;

using kmbLayersTestsConvolution = kmbLayersTests_nightly;

// NB: Kmb doesn't support compilation and inference on the same device
TEST_F(kmbLayersTestsConvolution, DISABLED_compilationLoadNetworkAndInfer) {
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
