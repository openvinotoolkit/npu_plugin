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

namespace vpu {
namespace HDDL2Plugin {

namespace IE = InferenceEngine;

class HDDL2Graph {
public:
    using Ptr = std::shared_ptr<HDDL2Graph>;

    virtual std::string getGraphName();
    virtual InferenceEngine::InputsDataMap getInputsInfo() const noexcept;
    virtual InferenceEngine::OutputsDataMap getOutputsInfo() const noexcept;
    virtual const std::string& getGraphBlob() const;

protected:
    std::string _graphName;
    std::string _blobContentString;
};

class HDDL2CompiledGraph : public HDDL2Graph {
public:
    using Ptr = std::shared_ptr<HDDL2CompiledGraph>;

    explicit HDDL2CompiledGraph(const IE::ICNNNetwork& network);
};

class HDDL2ImportedGraph : public HDDL2Graph {
public:
    using Ptr = std::shared_ptr<HDDL2ImportedGraph>;

    explicit HDDL2ImportedGraph(const std::string& blobFilename);

protected:
    std::string _blobFileName;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
