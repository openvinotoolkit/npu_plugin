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
#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv
{

    template <typename T_node, typename T_edge, typename T_node_iterator_comp, typename T_edge_iterator_comp>
    std::vector<typename graph<T_node, T_edge>::edge_list_iterator> critical_path(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator source, typename graph<T_node, T_edge>::node_list_iterator sink, const std::map<typename graph<T_node, T_edge>::node_list_iterator, unsigned, T_node_iterator_comp>& nodeCosts, std::map<typename graph<T_node, T_edge>::edge_list_iterator, unsigned, T_edge_iterator_comp> edgeCosts = std::map<typename graph<T_node, T_edge>::edge_list_iterator, unsigned, T_edge_iterator_comp>())
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        for(auto mapIt: nodeCosts)
        {
            auto nodeIt = mapIt.first;
            for(auto parentIt = nodeIt->leftmost_input(); parentIt != g.edge_end(); ++parentIt)
                edgeCosts[parentIt] += mapIt.second;
        }
        return dijkstra<T_node, T_edge, T_node_iterator_comp, T_edge_iterator_comp>(g, source, sink, edgeCosts);
    }

}

#endif // CRITICAL_PATH_HPP_
