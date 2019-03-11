#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"
#include "include/mcm/algorithms/transitive_reduction.hpp"

using graph_char_int= mv::graph<char, int>;

TEST (graph_transitive_reduction, test1)
{
    // Define graph
    graph_char_int g;

    auto na = g.node_insert('a');
    auto nb = g.node_insert('b');
    auto nc = g.node_insert('c');
    auto nd = g.node_insert('d');
    auto ne = g.node_insert('e');

    g.edge_insert(na, nb, 1);
    g.edge_insert(na, nc, 1);
    g.edge_insert(na, nd, 1);
    g.edge_insert(na, ne, 1);
    g.edge_insert(nb, nd, 1);
    g.edge_insert(nc, nd, 1);
    g.edge_insert(nc, ne, 1);
    g.edge_insert(nd, ne, 1);

    mv::transitiveReduction(g);

}
