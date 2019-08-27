#ifndef TRANSITIVE_REDUCTION_HPP_
#define TRANSITIVE_REDUCTION_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/topological_sort.hpp"
#include <map>
#include <vector>
#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv
{
    template <typename NodeIterator>
    struct OpItComparatorTemplate
    {
        bool operator()(NodeIterator lhs, NodeIterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };


    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComp, typename NodeItComp>
    void transitiveReduction_(graph<T_node, T_edge>& g,
                typename graph<T_node, T_edge>::node_list_iterator root,
                const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComp>& filteredEdges,
                std::set<typename graph<T_node, T_edge>::node_list_iterator, NodeItComp>& processedNodes)
    {
        // Collecting the set of neighbours, as edges
        // NOTE: Can't use unordered map because node_list_iterator needs to be hashable (requirement too strict)
        std::map<typename graph<T_node, T_edge>::node_list_iterator,
                typename graph<T_node, T_edge>::edge_list_iterator,
                OpItComparatorTemplate<typename graph<T_node, T_edge>::node_list_iterator>> root_adj, toEliminate;

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
            root_adj[e->sink()] = e;

        // Starting a DFS from each neighbour v
        // If a node u is reachable from v and it's also a neighbour of the root
        // Eliminate the edge between root and u

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            // Must skip first edge (itself)
            typename graph<T_node, T_edge>::edge_dfs_iterator edge_dfs(e);
            ++edge_dfs;
            for (; edge_dfs != g.edge_end(); ++edge_dfs)
            {
                auto u = edge_dfs->sink();
                auto it = root_adj.find(u);
                if(it != root_adj.end())
                {
                    auto it2 = filteredEdges.find(it->second);
                    if(it2 != filteredEdges.end())
                        continue;
                    toEliminate[u] = it->second;
                }
            }
        }

        for(auto edgeToEliminatePair : toEliminate)
            g.edge_erase(edgeToEliminatePair.second);

        processedNodes.insert(root);
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            auto v = e->sink();
            if (processedNodes.find(v) == processedNodes.end())
                transitiveReduction_(g, v, filteredEdges, processedNodes);
        }
    }

    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComparator, typename NodeItComparator>
    void transitiveReduction(graph<T_node, T_edge>& g,
                const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>&
                filteredEdges = std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>())
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        // Topological sort in this case also checks if the graph is dag. Hence, no explicit check for DAG is needed here.
        auto sortedNodes = topologicalSort(g);

        std::set<typename graph<T_node, T_edge>::node_list_iterator, NodeItComparator> processedNodes;

        for(auto node : sortedNodes)
        {
            if(node->parents_size() != 0)
                return;
            transitiveReduction_<T_node, T_edge, EdgeItComparator>(g, node, filteredEdges, processedNodes);
        }
    }
}

#endif
