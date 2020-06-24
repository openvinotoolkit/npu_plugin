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
#include <ie_data.h>
namespace vpux {

using DataMap = std::map<std::string, InferenceEngine::DataPtr>;
class NetworkDescription {
public:
    virtual DataMap getInputsInfo() = 0;
    virtual DataMap getOutputsInfo() = 0;
    virtual DataMap getRuntimeInputsInfo() = 0;
    virtual DataMap getRuntimeOutputsInfo() = 0;

    virtual std::vector<char> getCompiledNetwork() const = 0;
};

class Compiler {
public:
    virtual std::shared_ptr<NetworkDescription> compile(
        InferenceEngine::ICNNNetwork& network, const InferenceEngine::ParamMap params = {}) = 0;
    virtual std::shared_ptr<vpux::NetworkDescription> parse(const std::vector<char>& network) = 0;
    virtual std::set<std::string> getSupportedLayers(InferenceEngine::ICNNNetwork& network) = 0;
};

}
