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

#include <graph_tools.hpp>
#include <ade/typed_graph.hpp>
#include <ade/helpers/subgraphs.hpp>
#include <ade/util/filter_range.hpp>
#include <ade/util/iota_range.hpp>
#include <samples/slog.hpp>
#include <gflags/gflags.h>

namespace ade {
class Graph;
}  // namespace ade

using namespace InferenceEngine;

namespace Utils {
struct CNNLayerMetadata {
    CNNLayerPtr layer;

    static const char* name();
};

void translateNetworkToAde(ade::Graph& gr, ICNNNetwork& network);

struct modelPart {
    std::string path;
    details::CNNNetworkImplPtr networkPart;
};

void doSplitGraph(details::CNNNetworkImplPtr& clonedNetwork, std::map<std::string, modelPart>& requiredParts);

}  // namespace Utils
