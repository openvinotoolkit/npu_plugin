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

#include <utility>
#include <unordered_set>

namespace vpu {

template <typename NodeData>
class Node {
public:
    template <typename... Args>
    explicit Node(Args&&... args) : data{std::forward<Args>(args)...} {}

    ~Node() {
        for (auto&& predecessor : predecessors) {
            predecessor->removeSuccessor(*this);
        }
        for (auto&& successor : successors) {
            successor->removePredecessor(*this);
        }
    }

    const std::unordered_set<Node<NodeData>*>& getPredecessors() const {
        return predecessors;
    }

    const std::unordered_set<Node<NodeData>*>& getSuccessors() const {
        return successors;
    }

    const NodeData& getData() const {
        return data;
    }

    bool addPredecessor(Node<NodeData>& node) {
        return predecessors.insert(&node).second;
    }

    bool addSuccessor(Node<NodeData>& node) {
        return successors.insert(&node).second;
    }

    bool removePredecessor(Node<NodeData>& node) {
        return predecessors.erase(&node) == 1;
    }

    bool removeSuccessor(Node<NodeData>& node) {
        return successors.erase(&node) == 1;
    }

    std::size_t numPredecessors() const {
        return predecessors.size();
    }

    std::size_t numSuccessors() const {
        return successors.size();
    }

    bool isPredecessor(Node<NodeData>& node) const {
        return predecessors.find(&node) != predecessors.end();
    }

    bool isSuccessor(Node<NodeData>& node) const {
        return successors.find(&node) != successors.end();
    }

private:
    NodeData data;
    std::unordered_set<Node<NodeData>*> predecessors;
    std::unordered_set<Node<NodeData>*> successors;
};

template <typename NodeData>
void createEdge(Node<NodeData>& from, Node<NodeData>& to) {
    if (&from == &to) {
        return;
    }

    from.addSuccessor(to);
    to.addPredecessor(from);
}

}  // namespace vpu
