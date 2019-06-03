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

#include <list>
#include <memory>
#include <utility>

#include "details/ie_exception.hpp"

#include "node.hpp"

namespace vpu {

namespace details {

template <typename NodeData>
struct NodesInfo {
    std::list<Node<NodeData>> nodes;
    std::shared_ptr<NodesInfo> prev;
    bool isImmutable = false;
    std::size_t numNodes = 0;

    void makePrevImmutable() {
        if (!prev) {
            return;
        }
        prev->isImmutable = true;
    }
};

template <typename NodeData>
std::shared_ptr<NodesInfo<NodeData>> allocateNodesInfo(const std::shared_ptr<NodesInfo<NodeData>>& prev) {
    auto nodesInfo = std::make_shared<NodesInfo<NodeData>>();
    if (prev) {
        nodesInfo->prev = prev;
        nodesInfo->numNodes = prev->numNodes;
    }
    return nodesInfo;
}

template <typename NodeData, typename... Args>
Node<NodeData>& allocateNode(std::shared_ptr<NodesInfo<NodeData>>& info, Args&&... args) {
    assert(info);
    assert(!info->isImmutable);

    info->nodes.emplace_back(std::forward<Args>(args)...);
    ++info->numNodes;

    info->makePrevImmutable();

    return info->nodes.back();
}

}  // namespace details

template <class NodeData>
class GraphTree {
public:
    GraphTree() {
        nodes = std::make_shared<details::NodesInfo<NodeData>>();
    }

    GraphTree(const GraphTree& from) {
        assert(from.nodes);

        nodes = details::allocateNodesInfo(from.nodes);
    }

    GraphTree& operator=(GraphTree&) = delete;

    ~GraphTree() = default;

    template <typename... Args>
    Node<NodeData>& createNode(Args&&... args);

    std::size_t numNodes() const {
        return nodes->numNodes;
    }

private:
    void makeMutable() {
        assert(nodes);

        if (nodes->isImmutable) {
            nodes = details::allocateNodesInfo(nodes);
            assert(nodes);
        }
    }

    std::shared_ptr<details::NodesInfo<NodeData>> nodes;
};

template <class NodeData>
template <typename... Args>
Node<NodeData>& GraphTree<NodeData>::createNode(Args&&... args) {
    makeMutable();
    return details::allocateNode(nodes, std::forward<Args>(args)...);
}

}  // namespace vpu
