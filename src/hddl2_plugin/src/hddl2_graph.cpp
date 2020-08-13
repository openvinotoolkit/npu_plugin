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

#include <hddl2_exceptions.h>

#include <fstream>
#include <mcm_adapter.hpp>

#include "mcm_network_description.hpp"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
static std::vector<char> loadStreamToVector(std::istream& stream) {
    return std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

static std::vector<char> loadFileToVector(const std::string& filename) {
    std::ifstream streamWithBlob(filename, std::ios::binary);
    if (!streamWithBlob.is_open()) {
        THROW_IE_EXCEPTION << FILES_ERROR_str << "Could not open file: " << filename;
    }
    return loadStreamToVector(streamWithBlob);
}

static std::string extractFileName(const std::string& fullPath) {
    const size_t lastSlashIndex = fullPath.find_last_of("/\\");
    return fullPath.substr(lastSlashIndex + 1);
}

//------------------------------------------------------------------------------
vpux::NetworkDescription::Ptr Graph::compileGraph(InferenceEngine::ICNNNetwork& network, const vpu::MCMConfig& config) {
    // TODO We will throw exception of compilation, if not able to do that
    if (!MCMAdapter::isMCMCompilerAvailable()) {
        THROW_IE_EXCEPTION << "MCM compiler is not available!";
    }

    std::vector<char> graphBlob;
    try {
        MCMAdapter::compileNetwork(network, config, graphBlob);
    } catch (const std::exception& ex) {
        THROW_IE_EXCEPTION << "Failed to compile network! Error: " << ex.what();
    }
    return std::make_shared<MCMAdapter::MCMNetworkDescription>(graphBlob, config, network.getName());
}

vpux::NetworkDescription::Ptr Graph::importGraph(const std::string& blobFilename, const vpu::MCMConfig& config) {
    std::vector<char> blob = loadFileToVector(blobFilename);
    const std::string graphName = extractFileName(blobFilename);
    return std::make_shared<MCMAdapter::MCMNetworkDescription>(blob, config, graphName);
}

vpux::NetworkDescription::Ptr Graph::importGraph(std::istream& networkModel, const vpu::MCMConfig& config) {
    std::vector<char> blob = loadStreamToVector(networkModel);
    return std::make_shared<MCMAdapter::MCMNetworkDescription>(blob, config);
}
