//
// Copyright 2019 Intel Corporation.
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

#include <ie_blob.h>
#include <ie_layers.h>

#include <string>
#include <functional>
#include <unordered_set>
#include <vector>
#include <utility>
#include <details/caseless.hpp>

namespace NetworkSplitter {
using namespace InferenceEngine;
using namespace Utils;

using LayersSet = std::unordered_set<CNNLayerPtr>;

/// Split network on subgraphs based on layer affinity
///
/// @param network - source network
/// @param checkers - list of supported plugins
///
/// @return list of subgraphs
std::vector<LayersSet>
splitGraph(ICNNNetwork& network,
           const std::vector<std::string>& plugins);

/// Sort sugraphs topologically, behaviour is undefined if there are circular
/// refences between subgraps
///
/// @param subgraphs - list of subgraphs
void
sortSubgraphs(std::vector<LayersSet>& subgraphs);

}  // namespace NetworkSplitter
