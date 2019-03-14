#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"
#include "include/mcm/algorithms/topological_sort.hpp"

using graph_char_int= mv::graph<char, int>;
using graph_string_int= mv::graph<std::string, int>;

TEST (graph_topological_sort, test1)
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

    auto result = mv::topologicalSort(g);
    for(auto r: result)
        std::cout << *r << std::endl;

    // TODO: Write proper test with ASSERTS
    std::cout << "TOPOLOGICAL SORT ENDED" << std::endl;

}

TEST (graph_topological_sort, test2)
{
    // Define graph
    graph_string_int g;

    auto input = g.node_insert("Input");
    auto dma_input = g.node_insert("DMA_Input");
    auto convolution = g.node_insert("Convolution");
    auto weight = g.node_insert("Weight");
    auto dma_weight = g.node_insert("DMA_Weight");
    auto weight_table = g.node_insert("Weight_Table");
    auto dma_weight_table = g.node_insert("DMA_Weight_Table");
    auto sparsity_map = g.node_insert("Sparsity_map");
    auto dma_sparsity_map = g.node_insert("DMA_Sparsity_map");
    auto dma_output = g.node_insert("DMA_Output");
    auto output = g.node_insert("Output");

    g.edge_insert(input, dma_input, 1);
    g.edge_insert(dma_input, convolution, 1);
    g.edge_insert(weight, dma_weight, 1);
    g.edge_insert(weight_table, dma_weight_table, 1);
    g.edge_insert(sparsity_map, dma_sparsity_map, 1);
    g.edge_insert(dma_weight_table, convolution, 1);
    g.edge_insert(dma_sparsity_map, convolution, 1);
    g.edge_insert(convolution, dma_output, 1);
    g.edge_insert(dma_output, output, 1);

    auto result = mv::topologicalSort(g);
    for(auto r: result)
        std::cout << *r << std::endl;

    // TODO: Write proper test with ASSERTS
    std::cout << "TOPOLOGICAL SORT ENDED" << std::endl;

}
