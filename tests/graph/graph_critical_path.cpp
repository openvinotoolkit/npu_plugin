#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"
#include "include/mcm/algorithms/critical_path.hpp"

using graph_char_string = mv::graph<std::string, std::string>;

// NOTE: MUST FIND A GOOD TEST
// Example available at https://www.researchgate.net/figure/A-example-of-the-Dijkstra-algorithm_fig1_328036215
TEST (graph_critical_path, test1)
{
//    // Define graph
//    graph_char_string g;

//    auto n1 = g.node_insert("1");
//    auto n2 = g.node_insert("2");
//    auto n3 = g.node_insert("3");
//    auto n4 = g.node_insert("4");
//    auto n5 = g.node_insert("5");

//    g.edge_insert(n1, n2, "n1_n2");
//    g.edge_insert(n1, n4, "n1_n4");
//    g.edge_insert(n1, n5, "n1_n5");

//    g.edge_insert(n2, n3, "n2_n3");

//    g.edge_insert(n3, n5, "n3_n5");

//    g.edge_insert(n4, n3, "n4_n3");
//    g.edge_insert(n4, n5, "n4_n5");

//    std::map<std::string, int> costs;
//    costs[*n1] = 1;
//    costs[*n2] = 3;
//    costs[*n3] = 100;
//    costs[*n4] = 5;
//    costs[*n5] = 2;

//    auto result2 = mv::critical_path(g, n4, n5, costs);
//    ASSERT_EQ(*result2[0], "n4_n3");
//    ASSERT_EQ(*result2[1], "n3_n5");

//    std::cout << std::endl;

//    auto result1 = mv::critical_path(g, n1, n5, costs);
//    ASSERT_EQ(*result1[0], "n1_n4");
//    ASSERT_EQ(*result1[1], "n4_n3");
//    ASSERT_EQ(*result1[2], "n3_n5");

//    std::cout << std::endl;

}
