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

#pragma once

#include <map>
#include <memory>
#include <string>

#include "ie_core.hpp"
#include "ie_icnn_network.hpp"
#include "mcm_config.h"

namespace vpu {
namespace HDDL2Plugin {

class Graph {
public:
    using Ptr = std::shared_ptr<Graph>;

    std::string getGraphName() const { return _graphName; }
    std::string getGraphBlob() const { return _blobContentString; }

    InferenceEngine::InputsDataMap& getNetworkInputsInfo() noexcept { return _networkInputs; }
    InferenceEngine::OutputsDataMap& getNetworkOutputsInfo() noexcept { return _networkOutputs; }
    InferenceEngine::InputsDataMap& getDeviceInputsInfo() noexcept { return _deviceInputs; }
    InferenceEngine::OutputsDataMap& getDeviceOutputsInfo() noexcept { return _deviceOutputs; }

protected:
    std::string _graphName;
    std::string _blobContentString;

    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    InferenceEngine::InputsDataMap _deviceInputs;
    InferenceEngine::OutputsDataMap _deviceOutputs;

    void loadStreamToString(std::istream& model, std::string& outputString);
    void loadFileToString(const std::string& filename, std::string& outputString);
    std::string extractFileName(const std::string& fullPath);
    void getPortsFromBlob(const std::string& blobContentString, const MCMConfig& config);
};

class CompiledGraph : public Graph {
public:
    using Ptr = std::shared_ptr<CompiledGraph>;

    explicit CompiledGraph(InferenceEngine::ICNNNetwork& network, const MCMConfig& config);
};

class ImportedGraph : public Graph {
public:
    using Ptr = std::shared_ptr<ImportedGraph>;

    explicit ImportedGraph(std::istream& networkModel, const MCMConfig& config);
};

}  // namespace HDDL2Plugin
}  // namespace vpu
