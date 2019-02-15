#ifndef TRANSITIVE_REDUCTION_HPP_
#define TRANSITIVE_REDUCTION_HPP_

#include "include/mcm/graph/graph.hpp"
#include <map>
#include <vector>

namespace mv
{
    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge>
    void transitiveReduction(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator root)
    {
        // NOTE: what if T_node and T_edge are not hashable?
        // Let's try with maps, that require only operator <

        // Collecting the set of neighbours, as edges
        std::map<T_node, typename graph<T_node, T_edge>::edge_list_iterator> root_adj;
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
            root_adj[*(e->sink())] = e;

        // Starting a DFS from each neighbour v
        // If a node u is reachable from v and it's also a neighbour of the root
        // Eliminate the edge between root and u
        std::map<T_node, typename graph<T_node, T_edge>::edge_list_iterator> toEliminate;
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            auto v = e->sink();
            // Must skip first node (itself)
            typename graph<T_node, T_edge>::node_dfs_iterator u(v);
            ++u;
            for (; u != g.node_end(); ++u)
            {
                auto it = root_adj.find(*u);
                if(it != root_adj.end())
                    toEliminate[*u] = it->second;
            }
        }

        for(auto edgeToEliminatePair : toEliminate)
            g.edge_erase(edgeToEliminatePair.second);

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            auto v = e->sink();
            transitiveReduction(g, v);
        }
    }
}

#endif
