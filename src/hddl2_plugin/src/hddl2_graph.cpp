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

#include "blob_parser.hpp"
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
static std::string ExtractFileName(const std::string& fullPath) {
    const size_t lastSlashIndex = fullPath.find_last_of("/\\");
    return fullPath.substr(lastSlashIndex + 1);
}

ImportedGraph::ImportedGraph(const std::string& blobFilename, const MCMConfig& config) {
    // TODO find usage for mcmConfig in case of imported network
    UNUSED(config);

    std::ifstream blobFile(blobFilename, std::ios::binary);
    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << "[ERROR] *Could not open file: " << blobFilename;
    }

    std::ostringstream blobContentStream;
    blobContentStream << blobFile.rdbuf();
    const std::string& blobContentString = blobContentStream.str();

    _graphName = ExtractFileName(blobFilename);
    MCMAdapter::getNetworkInputs(blobContentString.c_str(), _networkInputs);
    MCMAdapter::getNetworkOutputs(blobContentString.c_str(), _networkOutputs);

    loadFileToString(blobFilename, _blobContentString);
}
