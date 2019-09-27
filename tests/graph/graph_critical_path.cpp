#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"
#include "include/mcm/algorithms/critical_path.hpp"

using graph_char_string = mv::graph<std::string, std::string>;

struct EdgeIteratorComp
{
    bool operator()(const graph_char_string::edge_list_iterator lhs, const graph_char_string::edge_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

struct NodeIteratorComp
{
    bool operator()(const graph_char_string::node_list_iterator lhs, const graph_char_string::node_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

// NOTE: MUST FIND A GOOD TEST
// Example available at https://www.researchgate.net/figure/A-example-of-the-Dijkstra-algorithm_fig1_328036215
