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

#include "mcm_network_description.hpp"

namespace vpu {
namespace HDDL2Plugin {

class Graph {
public:
    using Ptr = std::shared_ptr<Graph>;

    const std::string getGraphName() const { return _networkDescription->getName(); }
    const std::vector<char>& getGraphBlob() const { return _networkDescription->getCompiledNetwork(); }

    InferenceEngine::InputsDataMap getInputsInfo() noexcept {
        return MCMAdapter::helpers::dataMapIntoInputsDataMap(_networkDescription->getInputsInfo());
    }
    InferenceEngine::OutputsDataMap getOutputsInfo() noexcept {
        return MCMAdapter::helpers::dataMapIntoOutputsDataMap(_networkDescription->getOutputsInfo());
    }

protected:
    vpux::NetworkDescription::Ptr _networkDescription;

    std::string extractFileName(const std::string& fullPath);
};

class CompiledGraph : public Graph {
public:
    using Ptr = std::shared_ptr<CompiledGraph>;

    explicit CompiledGraph(InferenceEngine::ICNNNetwork& network, const vpu::MCMConfig& config);
};

class ImportedGraph : public Graph {
public:
    using Ptr = std::shared_ptr<ImportedGraph>;

    explicit ImportedGraph(const std::string& blobFilename, const vpu::MCMConfig& config);
    explicit ImportedGraph(std::istream& networkModel, const vpu::MCMConfig& config);
};

}  // namespace HDDL2Plugin
}  // namespace vpu
