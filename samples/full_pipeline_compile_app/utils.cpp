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

#include <ie_icnn_network.hpp>
#include <ie_util_internal.hpp>

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "utils.hpp"
#include <inference_engine.hpp>
#include "network_splitter.hpp"

using namespace InferenceEngine;
using namespace NetworkSplitter;

namespace Utils {
namespace {
using VisitedLayersMap = std::unordered_map<CNNLayer::Ptr, ade::NodeHandle>;
using TGraph = ade::TypedGraph<CNNLayerMetadata>;

std::vector<std::string> getAffinities(InferenceEngine::ICNNNetwork &network) {
    std::vector<std::string> ret;
    std::unordered_set<std::string> affinities;
    details::CNNNetworkIterator el(&network);

    while (el != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *el;
        if (!contains(affinities, layer->affinity)) {
           affinities.insert(layer->affinity);
           ret.push_back(layer->affinity);
        }
        el++;
    }

    return ret;
}

void translateVisitLayer(VisitedLayersMap& visited,
                TGraph& gr,
                const ade::NodeHandle& prevNode,
                const CNNLayer::Ptr& layer) {
    assert(nullptr != layer);;
    assert(!ade::util::contains(visited, layer));
    auto node = gr.createNode();
    gr.metadata(node).set(CNNLayerMetadata{layer});
    if (nullptr != prevNode) {
        gr.link(prevNode, node);
    }
    visited.insert({layer, node});
    for (auto&& data : layer->outData) {
        for (auto&& layerIt : data->getInputTo()) {
            auto nextLayer = layerIt.second;
            auto it = visited.find(nextLayer);
            if (visited.end() == it) {
                translateVisitLayer(visited, gr, node, nextLayer);
            } else {
                gr.link(node, it->second);
            }
        }
    }
}
}  // namespace

void translateNetworkToAde(ade::Graph& gr, ICNNNetwork& network) {
    TGraph tgr(gr);
    VisitedLayersMap visited;
    for (auto& data : getRootDataObjects(network)) {
        assert(nullptr != data);
        std::cout << "Root data: " << data->getName() << std::endl;
        for (auto& layerIt : data->getInputTo()) {
            auto layer = layerIt.second;
            std::cout << "\t\t layer: " << layer->name << std::endl;
            assert(nullptr != layer);
            if (!ade::util::contains(visited, layer)) {
                translateVisitLayer(visited, tgr, nullptr, layer);
            }
        }
    }
}
const char* CNNLayerMetadata::name() {
    return "CNNLayerMetadata";
}

void doSplitGraph(details::CNNNetworkImplPtr& clonedNetwork, std::map<std::string, modelPart>& requiredParts) {
    InputsDataMap externalInputsData;
    clonedNetwork->getInputsInfo(externalInputsData);

    OutputsDataMap externalOutputsData;
    clonedNetwork->getOutputsInfo(externalOutputsData);

    auto subgraphs = splitGraph(*clonedNetwork, /*affinities*/getAffinities(*clonedNetwork));

    sortSubgraphs(subgraphs);
    std::vector<CNNLayerPtr> tempLayers;

    InferenceEngine::ICNNNetworkStats* networkStats = nullptr;
    if (StatusCode::OK != clonedNetwork->getStats(&networkStats, nullptr)) {
        networkStats = nullptr;
    }

    for (auto &&subgraph : subgraphs) {
        auto affinity = (*subgraph.begin())->affinity;
        tempLayers.assign(subgraph.begin(), subgraph.end());
        requiredParts[affinity].networkPart = cloneNet(tempLayers, networkStats);
        (requiredParts[affinity].networkPart)->setName(clonedNetwork->getName() + "_" + affinity);
        // restoring some outputs from original net if they are not marked as output automatically
        // this might happen if output was set manually for origin network and
        // it doesn't go to next subgraph
        for (auto il : tempLayers) {
            if (externalOutputsData.find(il->name) != externalOutputsData.end()) {
                (requiredParts[affinity].networkPart)->addOutput(il->name);
            }
        }

        // update of pre-processing info

        InputsDataMap clonedInputs;
        (requiredParts[affinity].networkPart)->getInputsInfo(clonedInputs);
        for (auto &&it : externalInputsData) {
            auto inp = clonedInputs.find(it.first);
            if (inp != clonedInputs.end() && nullptr != inp->second) {
                inp->second->setPrecision(it.second->getPrecision());
                inp->second->getPreProcess() = it.second->getPreProcess();
            }
        }

        ResponseDesc resp;
        std::string xml;
        std::string bin;
        std::string fname = (requiredParts[affinity].networkPart)->getName();
        std::replace(fname.begin(), fname.end(), '/', '_');
        if (requiredParts[affinity].path.empty()) {
            xml = "./" + fname + ".xml";
            bin = "./" + fname + ".bin";
        } else {
            xml = requiredParts[affinity].path + "/" + fname + ".xml";
            bin = requiredParts[affinity].path + "/" + fname + ".bin";
        }
        (requiredParts[affinity].networkPart)->serialize(xml, bin, &resp);
    }
}

}  // namespace Utils

