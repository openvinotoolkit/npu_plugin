//
// Copyright 2018-2019 Intel Corporation.
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

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <set>

#include <cpp/ie_cnn_network.h>
#include <details/caseless.hpp>

#include <vpu/frontend/stage_builder.hpp>
#include <vpu/model/model.hpp>
#include <vpu/custom_layer.hpp>
#include <vpu/utils/enums.hpp>

namespace vpu {

namespace ie = InferenceEngine;

VPU_DECLARE_ENUM(LayersOrder,
    DFS,
    BFS)

class IeNetworkParser final {
//
// Public API
//
public:
    void clear();
    void checkNetwork(const ie::CNNNetwork& network);

    void parseNetworkBFS(const ie::CNNNetwork& network);
    void parseNetworkDFS(const ie::CNNNetwork& network);

    ie::InputsDataMap networkInputs;
    ie::OutputsDataMap networkOutputs;
    std::unordered_map<ie::DataPtr, ie::Blob::Ptr> constDatas;
    std::vector<ie::CNNLayerPtr> orderedLayers;
};

}  // namespace vpu
