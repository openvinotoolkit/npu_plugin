#ifndef CRITICAL_PATH_HPP_
#define CRITICAL_PATH_HPP_

#include <iostream>
#include <queue>
#include <vector>
#include <functional>
#include <set>
#include <algorithm>
#include <map>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/dijkstra.hpp"

namespace mv
{

    template <typename T_node, typename T_edge>
    std::vector<typename graph<T_node, T_edge>::edge_list_iterator> critical_path(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator source, typename graph<T_node, T_edge>::node_list_iterator sink, const std::map<T_node, int>& nodeCosts, std::map<T_edge, int> edgeCosts = std::map<T_edge, int>())
    {
        for(auto mapIt: nodeCosts)
        {
            auto nodeIt = g.node_find(mapIt.first);
            for(auto parentIt = nodeIt->leftmost_input(); parentIt != g.edge_end(); ++parentIt)
                edgeCosts[*parentIt] += mapIt.second;
        }
        return dijkstra(g, source, sink, edgeCosts);
    }

}

#endif // CRITICAL_PATH_HPP_
