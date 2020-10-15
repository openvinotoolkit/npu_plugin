#ifndef PATH_EXISTS_HPP_
#define PATH_EXISTS_HPP_

#include "include/mcm/graph/graph.hpp"
#include <map>
#include <vector>
#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv
{
    template <typename T_node, typename T_edge>
    bool pathExists(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator source, typename graph<T_node, T_edge>::node_list_iterator target)
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        for (typename graph<T_node, T_edge>::node_dfs_iterator it(source); it != g.node_end(); ++it)
        {
            if (*it == *target)
                return true;
        }
        return false;
    }
}

#endif
