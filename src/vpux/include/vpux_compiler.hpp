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
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_remote_context.hpp>
#include <set>

namespace vpux {

using DataMap = std::map<std::string, InferenceEngine::DataPtr>;
class NetworkDescription {
public:
    using Ptr = std::shared_ptr<NetworkDescription>;
    using CPtr = std::shared_ptr<const NetworkDescription>;

    virtual const std::string& getName() const = 0;
    virtual const DataMap& getInputsInfo() const = 0;
    virtual const DataMap& getOutputsInfo() const = 0;
    virtual const DataMap& getDeviceInputsInfo() const = 0;
    virtual const DataMap& getDeviceOutputsInfo() const = 0;

    virtual const std::vector<char>& getCompiledNetwork() const = 0;
};

class Compiler {
public:
    virtual std::shared_ptr<NetworkDescription> compile(
        InferenceEngine::ICNNNetwork& network, const InferenceEngine::ParamMap params = {}) = 0;
    virtual std::shared_ptr<vpux::NetworkDescription> parse(const std::vector<char>& network) = 0;
    virtual std::set<std::string> getSupportedLayers(InferenceEngine::ICNNNetwork& network) = 0;
};

}  // namespace vpux
