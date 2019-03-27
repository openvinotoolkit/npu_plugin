#ifndef EDGE_EXISTS_HPP_
#define EDGE_EXISTS_HPP_

#include "include/mcm/graph/graph.hpp"
#include <map>
#include <vector>

namespace mv
{
    template <typename T_node, typename T_edge>
    bool edgeExists(typename graph<T_node, T_edge>::node_list_iterator source, typename graph<T_node, T_edge>::node_list_iterator target, typename graph<T_node, T_edge>::node_list_iterator end)
    {
        for (typename graph<T_node, T_edge>::node_child_iterator it(source); it != end; ++it)
        {
            if (*it == *target)
                return true;
        }
        return false;
    }

    template <typename T_node, typename T_edge>
    bool edgeExists(graph<T_node, T_edge>& g, typename graph<T_node, T_edge>::node_list_iterator source, typename graph<T_node, T_edge>::node_list_iterator target)
    {
        return(edgeExists<T_node, T_edge>(source, target, g.node_end()));
    }
}

#endif
