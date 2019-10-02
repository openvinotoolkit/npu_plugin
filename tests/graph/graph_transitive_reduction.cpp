#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"
#include "include/mcm/algorithms/transitive_reduction.hpp"

using graph_string_string = mv::graph<std::string, std::string>;

struct EdgeItComparator
{
    bool operator()(graph_string_string::edge_list_iterator lhs, graph_string_string::edge_list_iterator rhs) const {
        return (*lhs) < (*rhs);
    }
};

struct NodeItComparator
{
    bool operator()(graph_string_string::node_list_iterator lhs, graph_string_string::node_list_iterator rhs) const {
        return (*lhs) < (*rhs);
    }
};

TEST (graph_transitive_reduction, test1)
{
    // Define graph
    graph_string_string g;

    auto na = g.node_insert("a");
    auto nb = g.node_insert("b");
    auto nc = g.node_insert("c");
    auto nd = g.node_insert("d");
    auto ne = g.node_insert("e");

    g.edge_insert(na, nb, "na_nb");
    g.edge_insert(na, nc, "na_nc");
    g.edge_insert(na, nd, "na_nd");
    g.edge_insert(na, ne, "na_ne");
    g.edge_insert(nb, nd, "nb_nd");
    g.edge_insert(nc, nd, "nc_nd");
    g.edge_insert(nc, ne, "nc_ne");
    g.edge_insert(nd, ne, "nd_ne");

    mv::transitiveReduction<std::string, std::string, EdgeItComparator, NodeItComparator>(g);

    // After transitive reduction, the edges removed should be
    // na_nd, na_ne, nc_ne

    std::set<std::string> eliminated_edges;
    eliminated_edges.insert("na_nb");
    eliminated_edges.insert("na_nc");
    eliminated_edges.insert("na_nd");
    eliminated_edges.insert("na_ne");
    eliminated_edges.insert("nb_nd");
    eliminated_edges.insert("nc_nd");
    eliminated_edges.insert("nc_ne");
    eliminated_edges.insert("nd_ne");

    for(auto e = g.edge_begin(); e != g.edge_end(); ++e)
        std::cout << eliminated_edges.erase(*e) << std::endl;

    ASSERT_EQ(eliminated_edges.size(), 3);
    ASSERT_EQ(eliminated_edges.count("na_nd"), 1);
    ASSERT_EQ(eliminated_edges.count("na_ne"), 1);
    ASSERT_EQ(eliminated_edges.count("nc_ne"), 1);

    std::cout << "Finished" << std::endl;
}
