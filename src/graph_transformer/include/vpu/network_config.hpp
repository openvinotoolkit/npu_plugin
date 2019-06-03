//
// Copyright (C) 2018-2019 Intel Corporation.
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

#include <vpu/graph_transformer.hpp>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <details/caseless.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class NetworkConfig final {
public:
    void parse(const CompilationConfig& config);

    bool skipAllLayers() const;
    bool skipLayerType(const std::string& layerType) const { return _noneLayers.count(layerType) != 0; }

    bool hasManualDataScale() const { return !_dataScale.empty(); }
    const std::unordered_map<std::string, float>& dataScale() const { return _dataScale; }

    bool hwDisabled(const std::string& layerName) const;

private:
    ie::details::caseless_set<std::string> _noneLayers;

    std::unordered_map<std::string, float> _dataScale;

    std::unordered_set<std::string> _hwWhiteList;
    std::unordered_set<std::string> _hwBlackList;
};

}  // namespace vpu
