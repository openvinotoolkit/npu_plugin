//
// Copyright 2020 Intel Corporation.
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

#include "hddl2_graph.h"

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "hddl2_exceptions.h"
#include "hddl2_helpers.h"
#include "mcm_adapter.hpp"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static void loadFileToString(const std::string& filename, std::string& outputString) {
    std::ifstream blobFile(filename, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << FILES_ERROR_str << "Could not open file: " << filename;
    }
    outputString = std::string(std::istreambuf_iterator<char>(blobFile), std::istreambuf_iterator<char>());
}

//------------------------------------------------------------------------------
//      class CompiledGraph Implementation
//------------------------------------------------------------------------------
CompiledGraph::CompiledGraph(IE::ICNNNetwork& network, const MCMConfig& config) {
    _graphName = network.getName();

    network.getInputsInfo(_networkInputs);
    network.getOutputsInfo(_networkOutputs);

    if (!MCMAdapter::isMCMCompilerAvailable()) {
        THROW_IE_EXCEPTION << "MCM compiler is not available!";
    }

    std::vector<char> graphBlob;
    try {
        MCMAdapter::compileNetwork(network, config, graphBlob);
    } catch (const std::exception& ex) {
        THROW_IE_EXCEPTION << "Failed to compile network! Error: " << ex.what();
    }
    _blobContentString = std::string(graphBlob.begin(), graphBlob.end());
}

//------------------------------------------------------------------------------
//      class ImportedGraph Implementation
//------------------------------------------------------------------------------
// TODO Hardcoded resnet until parsing imported blob will be supported #-25765
static InferenceEngine::InputsDataMap hardcodedResNetInputs() {
    InferenceEngine::InputsDataMap m_networkInputs;
    InferenceEngine::SizeVector inputDims({1, 3, 224, 224});
    InferenceEngine::Layout inputLayout = InferenceEngine::Layout::NCHW;
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc inputDesc(inputPrecision, inputDims, inputLayout);
    InferenceEngine::Data inputData("input", inputDesc);

    InferenceEngine::InputInfo inputInfo;
    inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
    m_networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    return m_networkInputs;
}

// TODO Hardcoded resnet until parsing imported blob will be supported #-25765
static InferenceEngine::OutputsDataMap hardcodedResNetOutputs() {
    InferenceEngine::OutputsDataMap m_networkOutputs;
    // TODO Hack for output to align with emulator
    InferenceEngine::SizeVector outputDims({1, 512, 1, 1});
    InferenceEngine::Layout outputLayout = InferenceEngine::Layout::NCHW;
    // TODO Should it be FP32?
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::U8;
    InferenceEngine::TensorDesc outputDesc(outputPrecision, outputDims, outputLayout);

    InferenceEngine::Data outputData("output", outputDesc);
    m_networkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    return m_networkOutputs;
}

ImportedGraph::ImportedGraph(const std::string& blobFilename, const MCMConfig& config) {
    // TODO find usage for mcmConfig in case of imported network
    UNUSED(config);

    // TODO Hardcoded resnet until parsing imported blob will be supported #-25765
    _graphName = "resNet";
    _networkInputs = hardcodedResNetInputs();
    _networkOutputs = hardcodedResNetOutputs();
    loadFileToString(blobFilename, _blobContentString);
}
