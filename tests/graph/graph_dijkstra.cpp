#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"
#include "include/mcm/algorithms/dijkstra.hpp"

using graph_char_string= mv::graph<std::string, std::string>;

struct NodeIteratorComp
{
    bool operator()(const graph_char_string::node_list_iterator lhs, const graph_char_string::node_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

struct EdgeIteratorComp
{
    bool operator()(const graph_char_string::edge_list_iterator lhs, const graph_char_string::edge_list_iterator rhs) const
    {
        return (*lhs) < (*rhs);
    }
};

//Example available at https://www.researchgate.net/figure/A-example-of-the-Dijkstra-algorithm_fig1_328036215
TEST (graph_dijkstra, test1)
{
    // Define graph
    graph_char_string g;

    auto n1 = g.node_insert("1");
    auto n2 = g.node_insert("2");
    auto n3 = g.node_insert("3");
    auto n4 = g.node_insert("4");
    auto n5 = g.node_insert("5");

    auto n1_n2 = g.edge_insert(n1, n2, "n1_n2");
    auto n1_n4 = g.edge_insert(n1, n4, "n1_n4");
    auto n1_n5 = g.edge_insert(n1, n5, "n1_n5");

    auto n2_n3 = g.edge_insert(n2, n3, "n2_n3");

    auto n3_n5 = g.edge_insert(n3, n5, "n3_n5");

    auto n4_n3 = g.edge_insert(n4, n3, "n4_n3");
    auto n4_n5 = g.edge_insert(n4, n5, "n4_n5");

    std::map<graph_char_string::edge_list_iterator, unsigned, EdgeIteratorComp> costs;
    costs[n1_n2] = 1;
    costs[n2_n3] = 5;
    costs[n4_n5] = 6;
    costs[n1_n5] = 100;
    costs[n1_n4] = 3;
    costs[n4_n3] = 2;
    costs[n3_n5] = 1;

    auto result2 = mv::dijkstra<std::string, std::string, NodeIteratorComp, EdgeIteratorComp>(g, n4, n5, costs);
    ASSERT_EQ(*result2[0], "n4_n3");
    ASSERT_EQ(*result2[1], "n3_n5");

    std::cout << std::endl;

    auto result1 = mv::dijkstra<std::string, std::string, NodeIteratorComp, EdgeIteratorComp>(g, n1, n5, costs);
    ASSERT_EQ(*result1[0], "n1_n4");
    ASSERT_EQ(*result1[1], "n4_n3");
    ASSERT_EQ(*result1[2], "n3_n5");

    std::cout << std::endl;

}
