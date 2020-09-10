#ifndef SCHEDULING_SORT
#define SCHEDULING_SORT

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/is_dag.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv
{
    template <typename T_node, typename T_edge>
    std::vector<typename graph<T_node, T_edge>::node_list_iterator> schedulingSort(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator initialNode)
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        std::vector<typename graph<T_node, T_edge>::node_list_iterator> toReturn;

        typename graph<T_node, T_edge>::node_bfs_iterator it(initialNode);

        for(; it != g.node_end(); ++it)
            toReturn.push_back(it);

        return toReturn;
    }
}

#endif
