#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/fathom/graph/graph.hpp"
#include "include/fathom/graph/stl_allocator.hpp"
#include "include/fathom/computation/model/serializer.hpp"

using int_int_graph = mv::graph<int, int, mv::stl_allocator>;
using char_int_graph = mv::graph<char, int, mv::stl_allocator>;

TEST (graph_seriallizer, blob_output) 
{

    int_int_graph utest_graph_00;

    // define test graph 0 
    int_int_graph::node_list_iterator itn0 = utest_graph_00.node_insert(800);
    int_int_graph::node_list_iterator itn1 ;
    itn1 = utest_graph_00.node_insert(801);
    auto itn7 = utest_graph_00.node_insert(807);
    int_int_graph::node_list_iterator itn2 = utest_graph_00.node_insert(itn1, 802, 90102);
    auto itn3 = utest_graph_00.node_insert(itn2, 803, 90203);
    int_int_graph::node_list_iterator itn5 ;
    itn5 = utest_graph_00.node_insert(itn3, 805, 90305);
    auto itn4 = utest_graph_00.node_insert(itn5, 804, 90504);
    auto itn6 = utest_graph_00.node_insert(itn5, 806, 90506);
    auto itn8 = utest_graph_00.node_insert(itn6, 808, 90608);
    int_int_graph::edge_list_iterator ite0402 = utest_graph_00.edge_insert(itn4, itn2, 90402);
    auto ite403 = utest_graph_00.edge_insert(itn4, itn3, 90403);
    auto ite405 = utest_graph_00.edge_insert(itn4, itn5, 90405);
    auto ite502 = utest_graph_00.edge_insert(itn5, itn2, 90502);
    auto ite604 = utest_graph_00.edge_insert(itn6, itn4, 90604);
    auto ite106 = utest_graph_00.edge_insert(itn1, itn6, 90106);
    auto ite701 = utest_graph_00.edge_insert(itn7, itn1, 90701);
    utest_graph_00.edge_insert(itn6, itn7, 90607);
    utest_graph_00.edge_insert(itn0, itn1, 90001);

    // declare serializer
    mv::Serializer gs(mv::mvblob_mode);

    // check blob output

/*
    FILE *obs;             // output file
    char const *out_file_name = "test_output.txt";

    if ((obs = fopen(out_file_name, "w")) == NULL)
    {
       printf("ERROR: Could not open output file \n");
    }
*/

    gs.write_blob(utest_graph_00);

    EXPECT_EQ (0,0) << "ERROR: unexpected blob output";

}
