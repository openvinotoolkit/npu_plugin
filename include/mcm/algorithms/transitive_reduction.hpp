#ifndef TRANSITIVE_REDUCTION_HPP_
#define TRANSITIVE_REDUCTION_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/topological_sort.hpp"
#include <map>
#include <vector>

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
    template <typename T_node, typename T_edge, typename EdgeItComp>
    void transitiveReduction_(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator root, const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComp>& filteredEdges)
    {
        // Collecting the set of neighbours, as edges
        std::map<typename graph<T_node, T_edge>::node_list_iterator, typename graph<T_node, T_edge>::edge_list_iterator, OpItComparatorTemplate<typename graph<T_node, T_edge>::node_list_iterator>> root_adj;
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
            root_adj[e->sink()] = e;

        // Starting a DFS from each neighbour v
        // If a node u is reachable from v and it's also a neighbour of the root
        // Eliminate the edge between root and u

        // NOTE: Can't use unordered map because node_list_iterator needs to be hashable (requirement too strict)
        std::map<typename graph<T_node, T_edge>::node_list_iterator, typename graph<T_node, T_edge>::edge_list_iterator, OpItComparatorTemplate<typename graph<T_node, T_edge>::node_list_iterator>> toEliminate;

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            // Must skip first node (itself)
            typename graph<T_node, T_edge>::edge_dfs_iterator u(e);
            for (; u != g.edge_end(); ++u)
            {
                auto it = root_adj.find(u->sink());
                auto it2 = filteredEdges.find(u);
                if(it != root_adj.end() && it2 == filteredEdges.end())
                    toEliminate[u->sink()] = it->second;
            }
        }

        for(auto edgeToEliminatePair : toEliminate)
            g.edge_erase(edgeToEliminatePair.second);

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            auto v = e->sink();
            transitiveReduction_(g, v, filteredEdges);
        }
    }

    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComparator>
    void transitiveReduction(graph<T_node, T_edge>& g, const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>& filteredEdges = std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>())
    {
        auto sortedNodes = topologicalSort(g);

        for(auto node : sortedNodes)
        {
            if(node->parents_size() != 0)
                return;
            transitiveReduction_<T_node, T_edge, EdgeItComparator>(g, node, filteredEdges);
        }
    }
}

#endif
