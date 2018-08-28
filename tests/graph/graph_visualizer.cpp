#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"

using int_int_graph = mv::graph<int, int>;
using char_int_graph = mv::graph<char, int>;

TEST (graph_visualizer, dot_output) 
{

    // build 4 test graphs
    int_int_graph utest_graph_00;
    int_int_graph utest_graph_01;
    char_int_graph utest_graph_02;
    int_int_graph utest_graph_03;    // empty container corner case

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

    // test graph 1
    auto root = utest_graph_01.node_insert(1);
    auto it = utest_graph_01.node_insert(root, 2, 10);
    auto it4 = utest_graph_01.node_insert(root, 3, 20);
    auto it2 = utest_graph_01.node_insert(it, 4, 30);
    it = utest_graph_01.node_insert(it, 5, 40);
    auto it3 = utest_graph_01.node_insert(it2, 6, 50);
    utest_graph_01.edge_insert(root, it2, 80);
    utest_graph_01.edge_insert(it2, it, 70);
    utest_graph_01.edge_insert(it, it3, 60);
    utest_graph_01.edge_insert(it3, root, 90);

    // test graph 2
    auto na = utest_graph_02.node_insert('a');
    auto nb = utest_graph_02.node_insert('b');
    auto nc = utest_graph_02.node_insert('c');
    auto nd = utest_graph_02.node_insert('d');
    auto ne = utest_graph_02.node_insert('e');
    auto nf = utest_graph_02.node_insert('f');
    auto ng = utest_graph_02.node_insert('g');
    auto nh = utest_graph_02.node_insert('h');
    auto ni = utest_graph_02.node_insert('i');
    auto nj = utest_graph_02.node_insert('j');
    auto nk = utest_graph_02.node_insert('k');
    auto nl = utest_graph_02.node_insert('l');
    auto nm = utest_graph_02.node_insert('m');
    auto nn = utest_graph_02.node_insert('n');
    auto no = utest_graph_02.node_insert('o');
    auto np = utest_graph_02.node_insert('p');
    auto e1 = utest_graph_02.edge_insert(na, nc, 1);
    auto e2 = utest_graph_02.edge_insert(na, nd, 2);
    auto e3 = utest_graph_02.edge_insert(nb, nd, 3);
    auto e4 = utest_graph_02.edge_insert(nb, ne, 4);
    auto e5 = utest_graph_02.edge_insert(nc, nd, 5);
    auto e6 = utest_graph_02.edge_insert(nc, nf, 6);
    auto e7 = utest_graph_02.edge_insert(nc, ng, 7);
    auto e8 = utest_graph_02.edge_insert(nd, ng, 8);
    auto e9 = utest_graph_02.edge_insert(ne, ng, 9);
    auto e10 = utest_graph_02.edge_insert(ne, nh, 10);
    auto e11 = utest_graph_02.edge_insert(nf, ng, 11);
    auto e12 = utest_graph_02.edge_insert(nf, ni, 12);
    auto e13 = utest_graph_02.edge_insert(ng, nj, 13);
    auto e14 = utest_graph_02.edge_insert(nh, nj, 14);
    auto e15 = utest_graph_02.edge_insert(nh, nk, 15);
    auto e16 = utest_graph_02.edge_insert(ni, nl, 16);
    auto e17 = utest_graph_02.edge_insert(nj, ni, 17);
    auto e18 = utest_graph_02.edge_insert(nj, nm, 18);
    auto e19 = utest_graph_02.edge_insert(nj, nn, 19);
    auto e20 = utest_graph_02.edge_insert(nk, nn, 20);
    auto e21 = utest_graph_02.edge_insert(nk, no, 21);
    auto e22 = utest_graph_02.edge_insert(nl, np, 22);
    auto e23 = utest_graph_02.edge_insert(nm, np, 23);
    auto e24 = utest_graph_02.edge_insert(nn, np, 24);
    auto e25 = utest_graph_02.edge_insert(no, np, 25);

    // expected dot format text output 
    const char *gold_string00 = "digraph G {\n800;\n800 -> 801 [ label = \"90001\" ];\n801;\n801 -> 802 [ label = \"90102\" ];\n801 -> 806 [ label = \"90106\" ];\n807;\n807 -> 801 [ label = \"90701\" ];\n802;\n802 -> 803 [ label = \"90203\" ];\n803;\n803 -> 805 [ label = \"90305\" ];\n805;\n805 -> 804 [ label = \"90504\" ];\n805 -> 806 [ label = \"90506\" ];\n805 -> 802 [ label = \"90502\" ];\n804;\n804 -> 802 [ label = \"90402\" ];\n804 -> 803 [ label = \"90403\" ];\n804 -> 805 [ label = \"90405\" ];\n806;\n806 -> 808 [ label = \"90608\" ];\n806 -> 804 [ label = \"90604\" ];\n806 -> 807 [ label = \"90607\" ];\n808;\n}\n" ;
    const char *gold_string01 = "digraph G {\n1;\n1 -> 2 [ label = \"10\" ];\n1 -> 3 [ label = \"20\" ];\n1 -> 4 [ label = \"80\" ];\n2;\n2 -> 4 [ label = \"30\" ];\n2 -> 5 [ label = \"40\" ];\n3;\n4;\n4 -> 6 [ label = \"50\" ];\n4 -> 5 [ label = \"70\" ];\n5;\n5 -> 6 [ label = \"60\" ];\n6;\n6 -> 1 [ label = \"90\" ];\n}\n" ;
    const char *gold_string02 = "digraph G {\na;\na -> c [ label = \"1\" ];\na -> d [ label = \"2\" ];\nb;\nb -> d [ label = \"3\" ];\nb -> e [ label = \"4\" ];\nc;\nc -> d [ label = \"5\" ];\nc -> f [ label = \"6\" ];\nc -> g [ label = \"7\" ];\nd;\nd -> g [ label = \"8\" ];\ne;\ne -> g [ label = \"9\" ];\ne -> h [ label = \"10\" ];\nf;\nf -> g [ label = \"11\" ];\nf -> i [ label = \"12\" ];\ng;\ng -> j [ label = \"13\" ];\nh;\nh -> j [ label = \"14\" ];\nh -> k [ label = \"15\" ];\ni;\ni -> l [ label = \"16\" ];\nj;\nj -> i [ label = \"17\" ];\nj -> m [ label = \"18\" ];\nj -> n [ label = \"19\" ];\nk;\nk -> n [ label = \"20\" ];\nk -> o [ label = \"21\" ];\nl;\nl -> p [ label = \"22\" ];\nm;\nm -> p [ label = \"23\" ];\nn;\nn -> p [ label = \"24\" ];\no;\no -> p [ label = \"25\" ];\np;\n}\n" ;
    const char *gold_string03 = "digraph G {\n}\n" ;

    // declare visualizer
    mv::Visualizer gv(mv::node_content, mv::edge_content);

//TODO reenable disjoint test after graph.is_disjoint() is fixed
//    gv.print_dot(utest_graph_00);
//    EXPECT_EQ (0, utest_graph_00.is_disjoint()) << "ERROR: graph is disjoint" << std::endl;

//    gv.print_dot(utest_graph_01);
//    EXPECT_EQ (0, utest_graph_01.is_disjoint()) << "ERROR: graph is disjoint" << std::endl;

//    gv.print_dot(utest_graph_02);
//    EXPECT_EQ (0, utest_graph_02.is_disjoint()) << "ERROR: graph is disjoint" << std::endl;

//    gv.print_dot(utest_graph_03);
//    EXPECT_EQ (0, utest_graph_03.is_disjoint()) << "ERROR: graph is disjoint" << std::endl;

    // check graph 0 dot output
    testing::internal::CaptureStdout();
    gv.print_dot(utest_graph_00);
    std::string output_string00 = testing::internal::GetCapturedStdout() ;

    EXPECT_STREQ (gold_string00, output_string00.c_str()) << "ERROR: unexpected dot output";

    // check graph 1 dot output
    testing::internal::CaptureStdout();
    gv.print_dot(utest_graph_01);
    std::string output_string01 = testing::internal::GetCapturedStdout() ;

    EXPECT_STREQ (gold_string01, output_string01.c_str()) << "ERROR: unexpected dot output";

    // check graph 2 dot output
    testing::internal::CaptureStdout();
    gv.print_dot(utest_graph_02);
    std::string output_string02 = testing::internal::GetCapturedStdout() ;

    EXPECT_STREQ (gold_string02, output_string02.c_str()) << "ERROR: unexpected dot output";

    // check graph 3 dot output
    testing::internal::CaptureStdout();
    gv.print_dot(utest_graph_03);
    std::string output_string03 = testing::internal::GetCapturedStdout() ;

    EXPECT_STREQ (gold_string03, output_string03.c_str()) << "ERROR: unexpected dot output";

}
