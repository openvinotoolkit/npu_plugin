#ifndef IS_DAG_HPP_
#define IS_DAG_HPP_

#include "include/mcm/graph/graph.hpp"
#include <stack>
#include <unordered_set>
#include <utility>
#include <vector>
#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv
{
    // isDAG() returns true iff the graph is acyclic.
    template <typename T_node, typename T_edge>
    bool isDAG(graph<T_node, T_edge>& g)
    {
        return getNodeInCycle(g).first;
    }

    template <typename T_node, typename T_edge>
    std::pair<bool, typename graph<T_node, T_edge>::node_list_iterator> getNodeInCycle(graph<T_node, T_edge>& g)
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)

        // To find a node within a cycle of a directed graph, we perform a
        // search of the graph, looking for back-links; if we find one, it has a
        // cycle, and the current node is part of it.
        //
        // Conceptually, we use three node states: "undiscovered", "discovered",
        // and "finished".
        //
        // Initially, all nodes are undiscovered.  To find a cycle, we
        // repeatedly explore the graph by selecting an undiscovered node and
        // attempting to find a cycle within the subgraph rooted at that node.
        // (Invariant: between explorations, no node is in the "discovered"
        // state.)
        //
        // To find a cycle from a node, we mark our initial node as discovered,
        // and use it to initialize a set of nodes remaining to be checked
        // within the current exploration, and then repeatedly select a node
        // from that set and iterate through its outgoing edges:
        //
        // * Edges to finished nodes are ignored; we know they do not lead to
        //   cycles.
        //
        // * Edges to discovered nodes indicate that we've found a back-link, a
        //   path back to a node previously discovered within the current
        //   depth-first exploration -- so we can return false.
        //
        // * Edges to undiscovered nodes indicate nodes that need to be marked
        //   as "discovered" and added to the current exploration.
        //
        // When the set of nodes to be checked is empty, the exploration is
        // complete, and all of the discovered nodes are marked "finished".
        //
        // When there are no remaining undiscovered nodes, the algorithm has
        // completed without finding a cycle.
        //
        //
        // Implementation notes:
        //
        // * We implement the algorithm iteratively to bound our stack
        //   consumption; recursion here has proven to be problematic when
        //   performing a search on larger graphs.
        //
        // * We maintain our node states externally, keeping the graph itself
        //   constant.
        //
        // * We maintain the set of nodes to be checked as a simple stack,
        //   giving us depth-first exploration semantics.

        // The set of finished nodes.
        std::unordered_set<size_t> finished;

        // The stack of nodes to be processed in the current exploration. At the
        // top of the loop over the pending stack, the second element is false
        // if the node's children need to be explored, and true if the node's
        // children have been explored (i.e. if we're unwinding).
        std::stack<std::pair<typename graph<T_node, T_edge>::node_list_iterator, bool>> pending;

        for (auto it = g.node_begin(); it; ++it)
        {
            if (finished.count(it->getID()))
            {
                continue;  // We know there's no cycle from here.
            }

            // Begin an exploration from the current node, looking for a cycle.

            std::unordered_set<size_t> discovered;
            pending.emplace(it, false);
            while (pending.size())
            {
                auto& current = pending.top().first;
                if (pending.top().second) {
                    discovered.erase(current->getID());
                    finished.insert(current->getID());
                    pending.pop();
                    continue;
                }
                pending.top().second = true;
                discovered.insert(current->getID());
                for (typename graph<T_node, T_edge>::node_child_iterator childIt(current); childIt; ++childIt)
                {
                    typename graph<T_node, T_edge>::node_list_iterator child = childIt;
                    if (finished.count(child->getID()))
                    {
                        // This child node is already in the finished set; no
                        // need to check it further.
                        continue;
                    }
                    if (discovered.count(child->getID()))
                    {
                        // The child node was already in the discovered set;
                        // we've found a cycle.
                        return std::make_pair(false, current);
                    }
                    pending.emplace(std::move(child), false);
                }
            }
        }

        // No cycles were found in this graph.
        return std::make_pair(true, g.node_end());
    }
}

#endif
