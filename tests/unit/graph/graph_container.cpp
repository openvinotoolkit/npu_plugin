#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <vector>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/visualizer.hpp"

using int_int_graph = mv::graph<int, int>;

TEST (graph_container, int_int_graph_contruction) {
    // build a test int_int_graph
    int_int_graph utest_graph_00;
  
    // insert isolated nodes 0,1,7 
    int_int_graph::node_list_iterator itn0 = utest_graph_00.node_insert(800);
    int_int_graph::node_list_iterator itn1 ;
    itn1 = utest_graph_00.node_insert(801);
    auto itn7 = utest_graph_00.node_insert(807);

    // insert nodes 2,3,5,4,6,8 with edges from 1,2,3,5,5,6 
    int_int_graph::node_list_iterator itn2 = utest_graph_00.node_insert(itn1, 802, 90102);
    auto itn3 = utest_graph_00.node_insert(itn2, 803, 90203);
    int_int_graph::node_list_iterator itn5 ; 
    itn5 = utest_graph_00.node_insert(itn3, 805, 90305);
    auto itn4 = utest_graph_00.node_insert(itn5, 804, 90504);
    auto itn6 = utest_graph_00.node_insert(itn5, 806, 90506);
    auto itn8 = utest_graph_00.node_insert(itn6, 808, 90608);

    // insert isolated edges 4-2 4-3 4-5 5-2 6-4 6-7 1-6 7-1 0-1
    int_int_graph::edge_list_iterator ite0402 = utest_graph_00.edge_insert(itn4, itn2, 90402);
    auto ite403 = utest_graph_00.edge_insert(itn4, itn3, 90403);
    auto ite405 = utest_graph_00.edge_insert(itn4, itn5, 90405);
    auto ite502 = utest_graph_00.edge_insert(itn5, itn2, 90502);
    auto ite604 = utest_graph_00.edge_insert(itn6, itn4, 90604);
    auto ite106 = utest_graph_00.edge_insert(itn1, itn6, 90106);
    auto ite701 = utest_graph_00.edge_insert(itn7, itn1, 90701);
    utest_graph_00.edge_insert(itn6, itn7, 90607);
    utest_graph_00.edge_insert(itn0, itn1, 90001);

    // check if int_int_graph is built correctly
    EXPECT_EQ (804, *itn4) << "ERROR: wrong node content referenced by node iterator";
    EXPECT_EQ (90502, *ite502) << "ERROR: wrong edge content referenced by edge iterator";
    EXPECT_EQ (9, utest_graph_00.node_size()) << "ERROR: wrong number of nodes" << std::endl;
    EXPECT_EQ (15, utest_graph_00.edge_size()) << "ERROR: wrong number of edges" << std::endl;

    const char *gold_string = "digraph G {\n800;\n800 -> 801 [ label = \"90001\" ];\n801;\n801 -> 802 [ label = \"90102\" ];\n801 -> 806 [ label = \"90106\" ];\n807;\n807 -> 801 [ label = \"90701\" ];\n802;\n802 -> 803 [ label = \"90203\" ];\n803;\n803 -> 805 [ label = \"90305\" ];\n805;\n805 -> 804 [ label = \"90504\" ];\n805 -> 806 [ label = \"90506\" ];\n805 -> 802 [ label = \"90502\" ];\n804;\n804 -> 802 [ label = \"90402\" ];\n804 -> 803 [ label = \"90403\" ];\n804 -> 805 [ label = \"90405\" ];\n806;\n806 -> 808 [ label = \"90608\" ];\n806 -> 804 [ label = \"90604\" ];\n806 -> 807 [ label = \"90607\" ];\n808;\n}\n" ;
    testing::internal::CaptureStdout();
    mv::Visualizer gv(mv::node_content, mv::edge_content);
    gv.print_dot(utest_graph_00);
    std::string output_string = testing::internal::GetCapturedStdout() ;

    EXPECT_STREQ (gold_string, output_string.c_str()) << "ERROR: unexpected dot output";

//----------- node children

    //  test node with 1 child
    int_int_graph::node_child_iterator child_n0(itn0);
    EXPECT_EQ (801, *child_n0) << "ERROR: wrong node referenced by node_child_iterator";
   
    // test node with multiple children 
    int test_sum_node_children = 0 ;
    for (int_int_graph::node_child_iterator chit4(itn4); chit4 != utest_graph_00.node_end(); ++chit4)
    {
        test_sum_node_children += *chit4 ;
    }

    EXPECT_EQ (2410, test_sum_node_children) << "ERROR: wrong nodes referenced by node_child_iterator";

    // test all nodes all children
    // 1259 = 801+802+806+801+803+805+802+804+806+802+803+805+807+804+808
    test_sum_node_children = 0 ;
    for (auto node_list_it = utest_graph_00.node_begin(); node_list_it != utest_graph_00.node_end(); ++node_list_it)
    {
        for (int_int_graph::node_child_iterator chitx(node_list_it); chitx != utest_graph_00.node_end(); ++chitx)
        {
            test_sum_node_children += *chitx ;
        }
    }
    EXPECT_EQ (12059, test_sum_node_children) << "ERROR: node_child_iterator ALL test";

//----------- node parents

    //  test node with 1 parent
    int_int_graph::node_parent_iterator parent_n1(itn1);
    EXPECT_EQ (800, *parent_n1) << "ERROR: wrong node referenced by node_parent_iterator";

    // test node with multiple parents 
    int test_sum_node_parents = 0 ;
    for (int_int_graph::node_parent_iterator parent_it5(itn5); parent_it5 != utest_graph_00.node_end(); ++parent_it5)
    {
        test_sum_node_parents += *parent_it5 ;
    }

    EXPECT_EQ (1607, test_sum_node_parents) << "ERROR: wrong nodes referenced by node_parent_iterator";

    // test all nodes all parents
    test_sum_node_parents = 0 ;
    for (int_int_graph::node_reverse_list_iterator node_tsil_it = utest_graph_00.node_rbegin(); node_tsil_it != utest_graph_00.node_rend(); ++node_tsil_it)
    {
        for (int_int_graph::node_parent_iterator parent_itx(node_tsil_it); parent_itx != utest_graph_00.node_end(); ++parent_itx)
        {
            test_sum_node_parents += *parent_itx ;
        }
    }
    EXPECT_EQ (12059, test_sum_node_parents) << "ERROR: node_parent_iterator ALL test";


//-------------- node siblings

    // test all nodes all siblings
    // 14482=s(n0)+s(n1)+s(n7)+s(n2)+s(n3)+s(5)+s(4)+s(n6)+s(n8)
    // 14482=0 + 0 + 804+808 + 803+804+805+806 + 802+805 + 802+803 + 807+802+806+80 + 802+804 + 807+804
    int test_sum_node_siblings = 0 ;

    for (auto node_list_it = utest_graph_00.node_begin(); node_list_it != utest_graph_00.node_end(); ++node_list_it)
    {
        for (int_int_graph::node_sibling_iterator sibitx(node_list_it); sibitx != utest_graph_00.node_end(); ++sibitx)
        {   
            test_sum_node_siblings += *sibitx ;
        }
    }
    EXPECT_EQ (14482, test_sum_node_siblings) << "ERROR: node_siblings_iterator ALL test ";

//------- edge children

    // test all edges all children
    int test_sum_edge_children = 0 ;

    for (auto edge_list_it = utest_graph_00.edge_begin(); edge_list_it != utest_graph_00.edge_end(); ++edge_list_it)
    {
       for (int_int_graph::edge_child_iterator itx(edge_list_it); itx != utest_graph_00.edge_end(); ++itx)
       {
            test_sum_edge_children += *itx ;
       }
    }

    EXPECT_EQ (2531418, test_sum_edge_children) << "ERROR: unexpected set of child edges";

//------- edge parents
 
    int test_sum_edge_parents = 0 ;

    for (auto edge_list_it = utest_graph_00.edge_begin(); edge_list_it != utest_graph_00.edge_end(); ++edge_list_it)
    {
       for (int_int_graph::edge_parent_iterator itx(edge_list_it); itx != utest_graph_00.edge_end(); ++itx)
       {
            test_sum_edge_parents += *itx ;
       }
    }

    EXPECT_EQ (2530913, test_sum_edge_parents) << "ERROR: unexpected set of parent edges";

//------- edge siblings 

    int test_sum_edge_siblings = 0 ;
    for (auto edge_list_it = utest_graph_00.edge_begin(); edge_list_it != utest_graph_00.edge_end(); ++edge_list_it)
    {
       for (int_int_graph::edge_sibling_iterator itx(edge_list_it); itx != utest_graph_00.edge_end(); ++itx)
       {
            test_sum_edge_siblings += *itx ;
       }
    }

    EXPECT_EQ (1809290, test_sum_edge_siblings) << "ERROR: unexpected set of sibling edges";

}

TEST (graph_container, search_order)
{

    int_int_graph g;

    auto root = g.node_insert(1);
    auto it = g.node_insert(root, 2, 10);
    auto it4 = g.node_insert(root, 3, 20);
    auto it2 = g.node_insert(it, 4, 30);
    it = g.node_insert(it, 5, 40);
    auto it3 = g.node_insert(it2, 6, 50);
    g.edge_insert(root, it2, 80);
    g.edge_insert(it2, it, 70);
    g.edge_insert(it, it3, 60);
    g.edge_insert(it3, root, 90);

//    g.print_dot();

    std::vector<int> node_list_order= {1,2,3,4,5,6 } ;
    int count_node_list = 0 ;
    for (auto nodeIt = g.node_begin(); nodeIt != g.node_end(); ++nodeIt)
    {
        EXPECT_EQ (node_list_order[count_node_list], *nodeIt) << "ERROR: wrong node traversing node list forward";
        count_node_list++; 
    }
    EXPECT_EQ (6, count_node_list) << "ERROR: wrong node count traversing node list forward";

    std::vector<int> node_reverse_order= { 6,5,4,3,2,1 } ;
    int count_node_reverse = 0 ;
    for (int_int_graph::node_reverse_list_iterator rit = g.node_rbegin(); rit != g.node_rend(); ++rit)
    {
        EXPECT_EQ (node_reverse_order[count_node_reverse], *rit) << "ERROR: wrong node traversing node list reverse";
        count_node_reverse++;
    }
    EXPECT_EQ (6, count_node_reverse) << "ERROR: wrong node count traversing node list reverse";

    std::vector<int> node_dfs_order= { 1,2,4,5,6,3 } ;
    int count_node_dfs = 0 ;
    for (int_int_graph::node_dfs_iterator dit = g.node_begin(); dit != g.node_end(); ++dit)
    {
        EXPECT_EQ (node_dfs_order[count_node_dfs], *dit) << "ERROR: wrong node order traversing depth first";
        count_node_dfs++;
    }
    EXPECT_EQ (6, count_node_dfs) << "ERROR: wrong node count during depth first search";

    std::vector<int> node_bfs_order= { 1,2,3,4,5,6 } ;
    int count_node_bfs = 0 ;
    for (int_int_graph::node_bfs_iterator bit = g.node_begin(); bit != g.node_end(); ++bit)
    {
        EXPECT_EQ (node_bfs_order[count_node_bfs], *bit) << "ERROR: wrong node order traversing breadth first";
        count_node_bfs++;
    }
    EXPECT_EQ (6, count_node_bfs) << "ERROR: wrong node count during breadth first search";

    int_int_graph g2;

    auto it00 = g2.node_insert(801);
    auto it02 = g2.node_insert(it00, 802, 90102);
    auto it03 = g2.node_insert(it02, 803, 90203);
    auto it04 = g2.node_insert(it02, 804, 90204);
    auto it05 = g2.node_insert(it02, 805, 90205);
    g2.edge_insert(it03, it02, 90302);
    g2.edge_insert(it04, it02, 90402);

    g2.node_erase(it02);

    std::vector<int> node_list_order2= {801,803,804,805 } ;
    int count_node_list2 = 0 ;
    for (auto nodeIt = g2.node_begin(); nodeIt != g2.node_end(); ++nodeIt)
    {
        EXPECT_EQ (node_list_order2[count_node_list2], *nodeIt) << "ERROR: wrong node order traversing node list forward";
        count_node_list2++;
    }
    EXPECT_EQ (4, count_node_list2) << "ERROR: wrong node count traversing node list forward";
}


TEST (graph_container, int_int_graph_manipulation) 
{

//----------  remove a node

    int_int_graph g2;

    auto it00 = g2.node_insert(801);
    auto it02 = g2.node_insert(it00, 802, 90102);
    auto it03 = g2.node_insert(it02, 803, 90203);
    auto it04 = g2.node_insert(it02, 804, 90204);
    auto it05 = g2.node_insert(it02, 805, 90205);
    g2.edge_insert(it03, it02, 90302);
    g2.edge_insert(it04, it02, 90402);

    g2.node_erase(it02);

    std::vector<int> node_list_order2= {801,803,804,805 } ;
    int count_node_list2 = 0 ;
    for (auto it = g2.node_begin(); it != g2.node_end(); ++it)
    {
        EXPECT_EQ (node_list_order2[count_node_list2], *it) << "ERROR: wrong node order traversing node list forward";
        count_node_list2++;
    }
    EXPECT_EQ (4, count_node_list2) << "ERROR: wrong node count traversing node list forward";


//--------- Corner cases

    // build a test int_int_graph for corner case test
    int_int_graph utest_graph_01;

    // insert isolated nodes 0,1,7 
    int_int_graph::node_list_iterator itn0 = utest_graph_01.node_insert(800);
    int_int_graph::node_list_iterator itn1 ;
    itn1 = utest_graph_01.node_insert(801);
    auto itn7 = utest_graph_01.node_insert(807);

    // insert nodes 2,3,5,4,6,8 with edges from 1,2,3,5,5,6 
    int_int_graph::node_list_iterator itn2 = utest_graph_01.node_insert(itn1, 802, 90102);
    auto itn3 = utest_graph_01.node_insert(itn2, 803, 90203);
    int_int_graph::node_list_iterator itn5 ;
    itn5 = utest_graph_01.node_insert(itn3, 805, 90305);
    auto itn4 = utest_graph_01.node_insert(itn5, 804, 90504);
    auto itn6 = utest_graph_01.node_insert(itn5, 806, 90506);
    auto itn8 = utest_graph_01.node_insert(itn6, 808, 90608);

    // insert isolated edges 4-2 4-3 4-5 5-2 6-4 6-7 1-6 7-1 0-1
    int_int_graph::edge_list_iterator ite0402 = utest_graph_01.edge_insert(itn4, itn2, 90402);
    auto ite403 = utest_graph_01.edge_insert(itn4, itn3, 90403);
    auto ite405 = utest_graph_01.edge_insert(itn4, itn5, 90405);
    auto ite502 = utest_graph_01.edge_insert(itn5, itn2, 90502);
    auto ite604 = utest_graph_01.edge_insert(itn6, itn4, 90604);
    auto ite106 = utest_graph_01.edge_insert(itn1, itn6, 90106);
    auto ite701 = utest_graph_01.edge_insert(itn7, itn1, 90701);
    utest_graph_01.edge_insert(itn6, itn7, 90607);
    utest_graph_01.edge_insert(itn0, itn1, 90001);

    // define a corner case
/*
    enum elem_type
        {
            node,
            edge,
            elem_type_end
        };

    enum iter_type
        {
            flist,
            rlist,
            parent,
            child,
            sibling,
            dfs,
            bfs
        };

    enum oper_type
        {
            add,
            remove,
            append,
            nop
        };

    enum time_type
        {
            before_search,
            during_start,
            during_middle,
            during_end,
            after_search
        };

    enum place_type
        {
            int_int_graph_begin,
            int_int_graph_middle,
            int_int_graph_end
        };


    class corner_case 
    {
        private:
            elem_type elem;        
            iter_type iter;
            oper_type oper;        
            time_type time;        
            place_type place;
 
        public:  
            corner_case(elem_type const_elem, iter_type const_iter, oper_type const_oper, time_type const_time, place_type const_place) 
            {
                elem = const_elem ;
                iter = const_iter ;
                oper = const_oper ;
                time = const_time ;
                place = const_place ;
            }

            void test_case(graph& test_graph)
            {
                std::cout << "testing corner case against int_int_graph " << std::endl;

                // traverse the int_int_graph using the chosen iterator, remove node
                for (graph::node_dfs_iterator dit = test_graph.node_begin(); dit != test_graph.node_end(); ++dit)
                {
 //                   if (*dit == 804) 
                    if (*dit == 801) 
                    {
                        std::cout << "removing node" << std::endl;                
                        test_graph.node_erase(dit);
                    }
                }
                std::cout << "done removing node" << std::endl;                

                // traverse the int_int_graph using the chosen iterator
                std::vector<int> n_dfs_order= { 800,801,806,808,804,805,802,803,807 } ;
                std::vector<int> nr4_dfs_order= { 800,801,806,808,805,802,803,807 } ;
                std::vector<int> nr1_dfs_order= { 800,806,808,804,805,802,803,807 } ;
                int cnt_node_dfs = 0 ;
                for (graph::node_dfs_iterator dit = test_graph.node_begin(); dit != test_graph.node_end(); ++dit)
                {
 //                   EXPECT_EQ (nr4_dfs_order[cnt_node_dfs], *dit) << "ERROR: wrong node order traversing depth first";
                    std::cout << "checking int_int_graph at node " << *dit << std::endl;                
                    EXPECT_EQ (nr1_dfs_order[cnt_node_dfs], *dit) << "ERROR: wrong node order traversing depth first";
                    cnt_node_dfs++;
                }

                EXPECT_EQ (8, cnt_node_dfs) << "ERROR: wrong node count during corner_case.test_case";
          
            }

            void print_case() 
            {
                std::cout << "Corner case is "<< elem <<" "<< iter <<" "<< oper <<" "<< time <<" "<< place << std::endl;                
            }
    };

    corner_case cc(edge,parent,add,before_search,graph_end);

    cc.print_case();

    for (int case_elemi=0; case_elemi!=elem_type_end; case_elemi++)
    {
        elem_type case_elem = static_cast<elem_type>(case_elemi);
        corner_case ccase(case_elem,parent,add,before_search,graph_end);
        std::cout << "loop count= "<< case_elemi << std::endl;                
        ccase.print_case();
        ccase.test_case(utest_graph_01);
    } 
*/
}

using char_int_graph = mv::graph<char, int>;

TEST (graph_container, int_int_graph_operations)
{

    /*
    digraph G {
        a -> c [ label = "1" ];
        a -> d [ label = "2" ];
        b -> d [ label = "3" ];
        b -> e [ label = "4" ];
        c -> d [ label = "5" ];
        c -> f [ label = "6" ];
        c -> g [ label = "7" ];
        d -> g [ label = "8" ];
        e -> g [ label = "9" ];
        e -> h [ label = "10" ];
        f -> g [ label = "11" ];
        f -> i [ label = "12" ];
        g -> j [ label = "13" ];
        h -> j [ label = "14" ];
        h -> k [ label = "15" ];
        i -> l [ label = "16" ];
        j -> i [ label = "17" ];
        j -> m [ label = "18" ];
        j -> n [ label = "19" ];
        k -> n [ label = "20" ];
        k -> o [ label = "21" ];
        l -> p [ label = "22" ];
        m -> p [ label = "23" ];
        n -> p [ label = "24" ];
        o -> p [ label = "25" ];
    }
    */

    // Define int_int_graph
    char_int_graph g;

    auto na = g.node_insert('a');
    auto nb = g.node_insert('b');
    auto nc = g.node_insert('c');
    auto nd = g.node_insert('d');
    auto ne = g.node_insert('e');
    auto nf = g.node_insert('f');
    auto ng = g.node_insert('g');
    auto nh = g.node_insert('h');
    auto ni = g.node_insert('i');
    auto nj = g.node_insert('j');
    auto nk = g.node_insert('k');
    auto nl = g.node_insert('l');
    auto nm = g.node_insert('m');
    auto nn = g.node_insert('n');
    auto no = g.node_insert('o');
    auto np = g.node_insert('p');

    auto e1 = g.edge_insert(na, nc, 1);
    auto e2 = g.edge_insert(na, nd, 2);
    auto e3 = g.edge_insert(nb, nd, 3);
    auto e4 = g.edge_insert(nb, ne, 4);
    auto e5 = g.edge_insert(nc, nd, 5);
    auto e6 = g.edge_insert(nc, nf, 6);
    auto e7 = g.edge_insert(nc, ng, 7);
    auto e8 = g.edge_insert(nd, ng, 8);
    auto e9 = g.edge_insert(ne, ng, 9);
    auto e10 = g.edge_insert(ne, nh, 10);
    auto e11 = g.edge_insert(nf, ng, 11);
    auto e12 = g.edge_insert(nf, ni, 12);
    auto e13 = g.edge_insert(ng, nj, 13);
    auto e14 = g.edge_insert(nh, nj, 14);
    auto e15 = g.edge_insert(nh, nk, 15);
    auto e16 = g.edge_insert(ni, nl, 16);
    auto e17 = g.edge_insert(nj, ni, 17);
    auto e18 = g.edge_insert(nj, nm, 18);
    auto e19 = g.edge_insert(nj, nn, 19);
    auto e20 = g.edge_insert(nk, nn, 20);
    auto e21 = g.edge_insert(nk, no, 21);
    auto e22 = g.edge_insert(nl, np, 22);
    auto e23 = g.edge_insert(nm, np, 23);
    auto e24 = g.edge_insert(nn, np, 24);
    auto e25 = g.edge_insert(no, np, 25);

    // Nodes and edges relations

    // Node intputs - for node nc
    std::vector<int> nodeg_inputs= { 7,8,9,11 } ;
    int count_nodeg_inputs = 0 ;
    for (auto it = ng->leftmost_input(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (nodeg_inputs[count_nodeg_inputs++], *it) << "ERROR: incorrect edge input to node g";
    }
    EXPECT_EQ (4, count_nodeg_inputs) << "ERROR: wrong number of input edges to node g";

    // Node outputs - for node nc
    std::vector<int> nodec_outputs= { 5,6,7 } ;
    int count_nodec_outputs = 0 ;
    for (auto it = nc->leftmost_output(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (nodec_outputs[count_nodec_outputs++], *it) << "ERROR: incorrect edge output from node c";
    }
    EXPECT_EQ (3, count_nodec_outputs) << "ERROR: wrong number of output edges from node c";

    // Edge source - for edge e1
    EXPECT_EQ ('a', *e1->source()) << "ERROR: wrong source node to edge 1";

    // Edge sink - for edge e1
    EXPECT_EQ ('c', *e1->sink()) << "ERROR: wrong sink node from edge 1";

    // Graph nodes traversing

    // List
    std::vector<int> nodes_list= { 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p' } ;
    int count_nodes_list = 0 ;
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_list[count_nodes_list++], *it) << "ERROR: incorrect node in list";
    }
    EXPECT_EQ (16, count_nodes_list) << "ERROR: wrong number of nodes in list";

    // Reverse list
    std::vector<int> nodes_rlist= { 'p','o','n','m','l','k','j','i','h','g','f','e','d','c','b','a'} ;
    int count_nodes_rlist = 0 ;
    for (char_int_graph::node_reverse_list_iterator it = g.node_rbegin(); it != g.node_rend(); ++it)
    {
        EXPECT_EQ (nodes_rlist[count_nodes_rlist++], *it) << "ERROR: incorrect node in reverse list";
    }
    EXPECT_EQ (16, count_nodes_rlist) << "ERROR: wrong number of nodes in reverse list";

    // Search
    auto dfs_fdir = char_int_graph::node_dfs_iterator::forward;
    auto dfs_rdir = char_int_graph::node_dfs_iterator::reverse;
    auto dfs_lside = char_int_graph::node_dfs_iterator::leftmost;
    auto dfs_rside = char_int_graph::node_dfs_iterator::rightmost;
    auto bfs_fdir = char_int_graph::node_bfs_iterator::forward;
    auto bfs_rdir = char_int_graph::node_bfs_iterator::reverse;
    auto bfs_lside = char_int_graph::node_bfs_iterator::leftmost;
    auto bfs_rside = char_int_graph::node_bfs_iterator::rightmost;

    // DFS (forward leftmost) - startng from node na
    std::vector<char> nodes_dfsla= { 'a','c','d','g','j','i','l','p','m','n','f' } ;
    int count_nodes_dfsla = 0 ;
    for (char_int_graph::node_dfs_iterator it(na); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_dfsla[count_nodes_dfsla++], *it) << "ERROR: incorrect node DFS left-node a";
    }
    EXPECT_EQ (11, count_nodes_dfsla) << "ERROR: wrong number of nodes ";

    // DFS (forward rightmost) - startng from node na
    std::vector<char> nodes_dfsra= { 'a','d','g','j','n','p','m','i','l','c','f','n' } ;
    int count_nodes_dfsra = 0 ;
    for (char_int_graph::node_dfs_iterator it(na, dfs_fdir, dfs_rside); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_dfsra[count_nodes_dfsra++], *it) << "ERROR: incorrect node DFS right node a";
    }
    EXPECT_EQ (11, count_nodes_dfsra) << "ERROR: wrong number of nodes in DFS right node a";

    // DFS (reverse leftmost) - startng from node np
    std::vector<char> nodes_dfsrlp= { 'p','l','i','f','c','a','j','g','d','b','e','h','m','n','k','o' } ;
    int count_nodes_dfsrlp = 0 ;
    for (char_int_graph::node_dfs_iterator it(np, dfs_rdir, dfs_lside); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_dfsrlp[count_nodes_dfsrlp++], *it) << "ERROR: incorrect node DFS-reverse left node p";
    }
    EXPECT_EQ (16, count_nodes_dfsrlp) << "ERROR: wrong number of nodes in DFS-reverse left node p";

    // DFS (reverse rigthmost)- startng from node np
    std::vector<char> nodes_dfsrrp= { 'p','o','k','h','e','b','n','j','g','f','c','a','d','m','l','i' } ;
    int count_nodes_dfsrrp = 0 ;
    for (char_int_graph::node_dfs_iterator it(np, dfs_rdir, dfs_rside); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_dfsrrp[count_nodes_dfsrrp++], *it) << "ERROR: incorrect node DFS-reverse right node p";
    }
    EXPECT_EQ (16, count_nodes_dfsrrp) << "ERROR: wrong number of nodes in DFS-reverse right node p";

    // BFS (forward leftmost) - startng from node na
    std::vector<char> nodes_bfsfla= { 'a','c','d','f','g','i','j','l','m','n','p' } ;
    int count_nodes_bfsfla = 0 ;
    for (char_int_graph::node_bfs_iterator it(na, bfs_fdir, bfs_lside); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_bfsfla[count_nodes_bfsfla++], *it) << "ERROR: incorrect node BFS-forward left node a";
    }
    EXPECT_EQ (11, count_nodes_bfsfla) << "ERROR: wrong number of nodes in BFS-forward left node a";

    // BFS (forward rightmost) - startng from node na
    std::vector<char> nodes_bfsfra= { 'a','d','c','g','f','j','i','n','m','l','p' } ;
    int count_nodes_bfsfra = 0 ;
    for (char_int_graph::node_bfs_iterator it(na, bfs_fdir, bfs_rside); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_bfsfra[count_nodes_bfsfra++], *it) << "ERROR: incorrect node BFS-forward right node a";
    }
    EXPECT_EQ (11, count_nodes_bfsfra) << "ERROR: wrong number of nodes in BFS-forward right node a";

    // BFS (reverse leftmost) - startng from node np
    std::vector<char> nodes_bfsrlp= { 'p','l','m','n','o','i','j','k','f','g','h','c','d','e','a','b' } ;
    int count_nodes_bfsrlp = 0 ;
    for (char_int_graph::node_bfs_iterator it(np, bfs_rdir, bfs_lside); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_bfsrlp[count_nodes_bfsrlp++], *it) << "ERROR: incorrect node BFS-reverse left node p";
    }
    EXPECT_EQ (16, count_nodes_bfsrlp) << "ERROR: wrong number of nodes in BFS-reverse left node p";

    // BFS (reverse rigthmost) - startng from node np
    std::vector<char> nodes_bfsrrp= { 'p','o','n','m','l','k','j','i','h','g','f','e','d','c','b','a' } ;
    int count_nodes_bfsrrp = 0 ;
    for (char_int_graph::node_bfs_iterator it(np, bfs_rdir, bfs_rside); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_bfsrrp[count_nodes_bfsrrp++], *it) << "ERROR: incorrect node BFS-reverse right node p";
    }
    EXPECT_EQ (16, count_nodes_bfsrrp) << "ERROR: wrong number of nodes in BFS-reverse right node p";

    // Graph edges traversing

    // List
    std::vector<int> edges_flist= { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25 } ;
    int count_edges_flist = 0 ;
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_flist[count_edges_flist++], *it) << "ERROR: incorrect edge in forward list";
    }
    EXPECT_EQ (25, count_edges_flist) << "ERROR: wrong number of edges in forward list";

    // Reverse list
    std::vector<int> edges_rlist= { 25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 } ;
    int count_edges_rlist = 0 ;
    for (char_int_graph::edge_reverse_list_iterator it = g.edge_rbegin(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_rlist[count_edges_rlist++], *it) << "ERROR: incorrect edge in reverse list";
    }
    EXPECT_EQ (25, count_edges_rlist) << "ERROR: wrong number of edges in reverse list";

    // DFS - startng from edge e1
    std::vector<int> edges_dfs1= { 1,5,8,13,17,16,22,18,23,19,24,6,11,12,7 } ;
    int count_edges_dfs1 = 0 ;
    for (char_int_graph::edge_dfs_iterator it(e1); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_dfs1[count_edges_dfs1++], *it) << "ERROR: incorrect edge in DFS edge 1";
    }
    EXPECT_EQ (15, count_edges_dfs1) << "ERROR: wrong number of edges in DFS edge 1";

    // BFS - starting from edge e1
    std::vector<int> edges_bfs1= { 1,5,6,7,8,11,12,13,16,17,18,19,22,23,24 } ;
    int count_edges_bfs1 = 0 ;
    for (char_int_graph::edge_bfs_iterator it(e1); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_bfs1[count_edges_bfs1++], *it) << "ERROR: incorrect edge in BFS edge 1";
    }
    EXPECT_EQ (15, count_edges_bfs1) << "ERROR: wrong number of edges in BFS edge 1";

    // Relationship nodes traversing

    // Chilren of node nc
    std::vector<char> nodes_chc= { 'd','f','g' } ;
    int count_nodes_chc = 0 ;
    for (char_int_graph::node_child_iterator it(nc); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_chc[count_nodes_chc++], *it) << "ERROR: incorrect child node of node c";
    }
    EXPECT_EQ (3, count_nodes_chc) << "ERROR: wrong number of children of node c";

    EXPECT_EQ ('d',  *nc->leftmost_child()) << "ERROR: wrong leftmost child node of node c";
    EXPECT_EQ ('g', *nc->rightmost_child()) << "ERROR: wrong rightmost child node of node c";

    // Parents of node np
    std::vector<char> nodes_pap= { 'l','m','n','o' } ;
    int count_nodes_pap= 0 ;
    for (char_int_graph::node_parent_iterator it(np); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_pap[count_nodes_pap++], *it) << "ERROR: incorrect parent of node p";
    }
    EXPECT_EQ (4, count_nodes_pap) << "ERROR: wrong number of parents of node p";

    EXPECT_EQ ('l', *np->leftmost_parent()) << "ERROR: wrong leftmost parent node of node p";
    EXPECT_EQ ('o', *np->rightmost_parent()) << "ERROR: wrong rightmost parent node of node p";

    // Siblings of node nd
    std::vector<char> nodes_sid= { 'c','e','f','g' } ;
    int count_nodes_sid= 0 ;
    for (char_int_graph::node_sibling_iterator it(nd); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_sid[count_nodes_sid++], *it) << "ERROR: incorrect sibling of node d";
    }
    EXPECT_EQ (4, count_nodes_sid) << "ERROR: wrong number of siblings of node d";

    EXPECT_EQ ('c', *nd->leftmost_sibling()) << "ERROR: wrong leftmost sibling node of node d";
    EXPECT_EQ ('g', *nd->rightmost_sibling()) << "ERROR: wrong rightmost sibling node of node d";

    // Relationship edges traversing

    // Chilren of edge e1
    std::vector<int> edges_ch1 = { 5,6,7 } ;
    int count_edges_ch1= 0 ;
    for (char_int_graph::edge_child_iterator it(e1); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_ch1[count_edges_ch1++], *it) << "ERROR: incorrect child of edge 1";
    }
    EXPECT_EQ (3, count_edges_ch1) << "ERROR: wrong number of children of edge 1";

    EXPECT_EQ (5, *e1->leftmost_child()) << "ERROR: wrong leftmost child edge of edge 1";
    EXPECT_EQ (7, *e1->rightmost_child()) << "ERROR: wrong rightmost child edge of edge 1";

    // Parents of edge e13
    std::vector<int> edges_pa13 = { 7,8,9,11 } ;
    int count_edges_pa13= 0 ;
    for (char_int_graph::edge_parent_iterator it(e13); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_pa13[count_edges_pa13++], *it) << "ERROR: incorrect child of edge 13";
    }
    EXPECT_EQ (4, count_edges_pa13) << "ERROR: wrong number of parents of edge 13";

    EXPECT_EQ (7, *e13->leftmost_parent()) << "ERROR: wrong leftmost parent edge of edge 13";
    EXPECT_EQ (11, *e13->rightmost_parent()) << "ERROR: wrong rightmost parent edge of edge 13";

    // Siblings of edge e17
    std::vector<int> edges_si17 = { 18,19 } ;
    int count_edges_si17= 0 ;
    for (char_int_graph::edge_sibling_iterator it(e17); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_si17[count_edges_si17++], *it) << "ERROR: incorrect sibling of edge 17";
    }
    EXPECT_EQ (2, count_edges_si17) << "ERROR: wrong number of siblings of edge 17";

    EXPECT_EQ (18, *e17->leftmost_sibling()) << "ERROR: wrong leftmost sibling edge of edge 17";
    EXPECT_EQ (19, *e17->rightmost_sibling()) << "ERROR: wrong rightmost sibling edge of edge 17";

    // Node removal (ng)
     
    // Initialize traversing before deletion
    char_int_graph::node_dfs_iterator dfs_nc(nc);
    char_int_graph::node_bfs_iterator bfs_nc(nc);
    char_int_graph::edge_dfs_iterator dfs_e4(e4);
    char_int_graph::edge_bfs_iterator bfs_e4(e4);

    // Delete ng
    g.node_erase(ng);

    // Check if iterators invalid
    EXPECT_EQ (ng, g.node_end()) << "ERROR: node g was not removed properly";
    EXPECT_EQ (e7, g.edge_end()) << "ERROR: edge 7 was not removed properly";
    EXPECT_EQ (e8, g.edge_end()) << "ERROR: edge 8 was not removed properly";
    EXPECT_EQ (e9, g.edge_end()) << "ERROR: edge 9 was not removed properly";
    EXPECT_EQ (e11, g.edge_end()) << "ERROR: edge 11 was not removed properly";

    // Nodes list
    std::vector<char> nodes_flnog = { 'a','b','c','d','e','f','h','i','j','k','l','m','n','o','p' } ;
    int count_nodes_flnog= 0 ;
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        EXPECT_EQ (nodes_flnog[count_nodes_flnog++], *it) << "ERROR: incorrect nodei in forward list after removal of node g";
    }
    EXPECT_EQ (15, count_nodes_flnog) << "ERROR: wrong number of nodes in forward list after removal of node g";

    // Node reverse list
    std::vector<char> nodes_rlnog = { 'p','o','n','m','l','k','j','i','h','f','e','d','c','b','a' } ;
    int count_nodes_rlnog= 0 ;
    for (char_int_graph::node_reverse_list_iterator it = g.node_rbegin(); it != g.node_rend(); ++it)
    {
        EXPECT_EQ (nodes_rlnog[count_nodes_rlnog++], *it) << "ERROR: incorrect node in reverse list after removal of node g";
    }
    EXPECT_EQ (15, count_nodes_rlnog) << "ERROR: wrong number of nodes in reverse list after removal of node g";

    // Edges list
    std::vector<int> edges_flnog = { 1,2,3,4,5,6,10,12,14,15,16,17,18,19,20,21,22,23,24,25  } ;
    int count_edges_flnog= 0 ;
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_flnog[count_edges_flnog++], *it) << "ERROR: incorrect edge in forward list after removal of node g";
    }
    EXPECT_EQ (20, count_edges_flnog) << "ERROR: wrong number of edges in forward list after removal of node g";

    // Edges reverse list
    std::vector<int> edges_rlnog = { 25,24,23,22,21,20,19,18,17,16,15,14,12,10,6,5,4,3,2,1 } ;
    int count_edges_rlnog= 0 ;
    for (char_int_graph::edge_reverse_list_iterator it = g.edge_rbegin(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (edges_rlnog[count_edges_rlnog++], *it) << "ERROR: incorrect edge in reverse list after removal of node g";
    }
    EXPECT_EQ (20, count_edges_rlnog) << "ERROR: wrong number of edges in reverse list after removal of node g";

    // DFS - startng from node nc
    std::vector<char> nodes_dfsnog = { 'c','d','f','i','l','p' } ;
    int count_nodes_dfsnog= 0 ;
    for (; dfs_nc != g.node_end(); ++dfs_nc)
    {
        EXPECT_EQ (nodes_dfsnog[count_nodes_dfsnog++], *dfs_nc) << "ERROR: incorrect node in DFS(c) after removal of node g";
    }
    EXPECT_EQ (6, count_nodes_dfsnog) << "ERROR: wrong number of nodes in DFS(c) after removal of node g";


    // BFS - starting from node nc
    std::vector<char> nodes_bfsnog = { 'c','d','f','i','l','p' } ;
    int count_nodes_bfsnog= 0 ;
    for (; bfs_nc != g.node_end(); ++bfs_nc)
    {
        EXPECT_EQ (nodes_bfsnog[count_nodes_bfsnog++], *bfs_nc) << "ERROR: incorrect node in BFS(c) after removal of node g";
    }
    EXPECT_EQ (6, count_nodes_dfsnog) << "ERROR: wrong number of nodes in BFS(c) after removal of node g";

    // DFS - startng from edge e4
    std::vector<int> edges_dfsnog = { 4,10,14,17,16,22,18,23,19,24,15,20,21,25 } ;
    int count_edges_dfsnog= 0 ;
    for (; dfs_e4 != g.edge_end(); ++dfs_e4)
    {
        EXPECT_EQ (edges_dfsnog[count_edges_dfsnog++], *dfs_e4) << "ERROR: incorrect edge in DFS(4) after removal of node g";
    }
    EXPECT_EQ (14, count_edges_dfsnog) << "ERROR: wrong number of edges in DFS(4) after removal of node g";

    // BFS - starting from edge e4
    std::vector<int> edges_bfsnog = { 4,10,14,15,17,18,19,20,21,16,23,24,25,22 } ;
    int count_edges_bfsnog= 0 ;
    for (; bfs_e4 != g.edge_end(); ++bfs_e4)
    {
        EXPECT_EQ (edges_bfsnog[count_edges_bfsnog++], *bfs_e4) << "ERROR: incorrect edge in BFS(4) after removal of node g";
    }
    EXPECT_EQ (14, count_edges_bfsnog) << "ERROR: wrong number of edges in BFS(4) after removal of node g";

    // Chilren of node ne
    std::vector<char> children_enog = { 'h' } ;
    int count_children_enog= 0 ;
    for (char_int_graph::node_child_iterator it(ne); it != g.node_end(); ++it)
    {
        EXPECT_EQ (children_enog[count_children_enog++], *it) << "ERROR: incorrect child of node e after removal of node g";
    }
    EXPECT_EQ (1, count_children_enog) << "ERROR: wrong number of children of node e after removal of node g";

    // Parents of node nj
    std::vector<char> parents_jnog = { 'h' } ;
    int count_parents_jnog= 0 ;
    for (char_int_graph::node_parent_iterator it(nj); it != g.node_end(); ++it)
    {
        EXPECT_EQ (parents_jnog[count_parents_jnog++], *it) << "ERROR: incorrect parent of node j after removal of node g";
    }
    EXPECT_EQ (1, count_parents_jnog) << "ERROR: wrong number of parents of node j after removal of node g";

    // Siblings of node nh
    std::vector<char> siblings_hnog = {  } ;
    int count_siblings_hnog= 0 ;
    for (char_int_graph::node_sibling_iterator it(nh); it != g.node_end(); ++it)
    {
        EXPECT_EQ (siblings_hnog[count_siblings_hnog++], *it) << "ERROR: incorrect sibling of node h after removal of node g";
    }
    EXPECT_EQ (0, count_siblings_hnog) << "ERROR: wrong number of siblings of node h after removal of node g";

    // Chilren of edge e5
    std::vector<int> children_5nog = {  } ;
    int count_children_5nog = 0 ;
    for (char_int_graph::edge_child_iterator it(e5); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (children_5nog[count_children_5nog++], *it) << "ERROR: incorrect child of edge 5 after removal of node g";
    }
    EXPECT_EQ (0, count_children_5nog) << "ERROR: wrong number of children of edge 5 fter removal of node g";

    // Parents of edge e18
    std::vector<int> parents_18nog = { 14 } ;
    int count_parents_18nog  = 0 ;
    for (char_int_graph::edge_parent_iterator it(e18); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (parents_18nog[count_parents_18nog++], *it) << "ERROR: incorrect parent of edge 18 after removal of node g";
    }
    EXPECT_EQ (1, count_parents_18nog) << "ERROR: wrong number of parents of edge 18 after removal of node g";

    // Siblings of edge e12
    std::vector<int> siblings_12nog = { } ;
    int count_siblings_12nog  = 0 ;
    for (char_int_graph::edge_sibling_iterator it(e12); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (siblings_12nog[count_siblings_12nog++], *it) << "ERROR: incorrect sibling of edge 12 after removal of node g";
    }
    EXPECT_EQ (0, count_siblings_12nog) << "ERROR: wrong number of siblings of edge 12 after removal of node g";

    // Edge removal (e14)
     
    // Initialize traversing before deletion
    char_int_graph::node_dfs_iterator dfs_nb(nb);
    char_int_graph::node_bfs_iterator bfs_nb(nb);
    dfs_e4 = e4;    
    bfs_e4 = e4;

    // Delete e14
    g.edge_erase(e14);

    // Check if iterator invalid
    EXPECT_EQ (g.edge_end(), e14) << "ERROR: edge 14 not removed";

    // Nodes list
    std::vector<char> flist_no14 = { 'a','b','c','d','e','f','h','i','j','k','l','m','n','o','p' } ;
    int count_flist_no14  = 0 ;
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        EXPECT_EQ (flist_no14[count_flist_no14++], *it) << "ERROR: incorrect node in forward list after removal of edge 14";
    }
    EXPECT_EQ (15, count_flist_no14) << "ERROR: wrong number of nodes in forward list after removal of edge 14";

    // Node reverse list
    std::vector<char> rlist_no14 = { 'p','o','n','m','l','k','j','i','h','f','e','d','c','b','a' } ;
    int count_rlist_no14  = 0 ;
    for (char_int_graph::node_reverse_list_iterator it = g.node_rbegin(); it != g.node_rend(); ++it)
    {
        EXPECT_EQ (rlist_no14[count_rlist_no14++], *it) << "ERROR: incorrect node in reverse list after removal of edge 14";
    }
    EXPECT_EQ (15, count_rlist_no14) << "ERROR: wrong number of nodes in reverse list after removal of edge 14";

    // Edges list
    std::vector<int> eflist_no14 = { 1,2,3,4,5,6,10,12,15,16,17,18,19,20,21,22,23,24,25 } ;
    int count_eflist_no14  = 0 ;
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (eflist_no14[count_eflist_no14++], *it) << "ERROR: incorrect edge in forward list after removal of edge 14";
    }
    EXPECT_EQ (19, count_eflist_no14) << "ERROR: wrong number of edges in forward list after removal of edge 14";

    // Edges reverse list
    std::vector<int> rflist_no14 = { 25,24,23,22,21,20,19,18,17,16,15,12,10,6,5,4,3,2,1 } ;
    int count_rflist_no14  = 0 ;
    for (char_int_graph::edge_reverse_list_iterator it = g.edge_rbegin(); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (rflist_no14[count_rflist_no14++], *it) << "ERROR: incorrect edge in forward list after removal of edge 14";
    }
    EXPECT_EQ (19, count_rflist_no14) << "ERROR: wrong number of edges in forward list after removal of edge 14";

    // DFS - startng from node nb
    std::vector<char> dfsb_no14 = { 'b','d','e','h','k','n','p','o' } ;
    int count_dfsb_no14  = 0 ;
    for (; dfs_nb != g.node_end(); ++dfs_nb)
    {
        EXPECT_EQ (dfsb_no14[count_dfsb_no14++], *dfs_nb) << "ERROR: incorrect node in DFS(b) after removal of edge 14";
    }
    EXPECT_EQ (8, count_dfsb_no14) << "ERROR: wrong number of nodes in DFS(b) after removal of edge 14";

    // BFS - starting from node nb
    std::vector<char> bfsb_no14 = { 'b','d','e','h','k','n','o','p' } ;
    int count_bfsb_no14  = 0 ;
    for (; bfs_nb != g.node_end(); ++bfs_nb)
    {
        EXPECT_EQ (bfsb_no14[count_bfsb_no14++], *bfs_nb) << "ERROR: incorrect node in BFS(b) after removal of edge 14";
    }
    EXPECT_EQ (8, count_dfsb_no14) << "ERROR: wrong number of nodes in DFS(b) after removal of edge 14";

    // DFS - startng from edge e4
    std::vector<int> dfs4_no14 = { 4,10,15,20,24,21,25 } ;
    int count_dfs4_no14  = 0 ;
    for (; dfs_e4 != g.edge_end(); ++dfs_e4)
    {
        EXPECT_EQ (dfs4_no14[count_dfs4_no14++], *dfs_e4) << "ERROR: incorrect edge in DFS(4) after removal of edge 14";
    }
    EXPECT_EQ (7, count_dfs4_no14) << "ERROR: wrong number of edges in DFS(4) after removal of edge 14";

    // BFS - starting from edge e4
    std::vector<int> bfs4_no14 = { 4,10,15,20,21,24,25 } ;
    int count_bfs4_no14  = 0 ;
    for (; bfs_e4 != g.edge_end(); ++bfs_e4)
    {
        EXPECT_EQ (bfs4_no14[count_bfs4_no14++], *bfs_e4) << "ERROR: incorrect edge in BFS(4) after removal of edge 14";
    }
    EXPECT_EQ (7, count_bfs4_no14) << "ERROR: wrong number of edges in BFS(4) after removal of edge 14";

    // Chilren of node nh
    std::vector<char> chh_no14 = { 'k' } ;
    int count_chh_no14  = 0 ;
    for (char_int_graph::node_child_iterator it(nh); it != g.node_end(); ++it)
    {
        EXPECT_EQ (chh_no14[count_chh_no14++], *it) << "ERROR: incorrect child of node h after removal of edge 14";
    }
    EXPECT_EQ (1, count_chh_no14) << "ERROR: wrong number of children of node h after removal of edge 14";

    // Parents of node nj
    std::vector<char> parh_no14 = {  } ;
    int count_parh_no14  = 0 ;
    for (char_int_graph::node_parent_iterator it(nj); it != g.node_end(); ++it)
    {
        EXPECT_EQ (parh_no14[count_parh_no14++], *it) << "ERROR: incorrect parent of node h after removal of edge 14";
    }
    EXPECT_EQ (0, count_parh_no14) << "ERROR: wrong number of parents of node h after removal of edge 14";

    // Siblings of node nk
    std::vector<char> sibk_no14 = {  } ;
    int count_sibk_no14  = 0 ;
    for (char_int_graph::node_sibling_iterator it(nk); it != g.node_end(); ++it)
    {
        EXPECT_EQ (sibk_no14[count_sibk_no14++], *it) << "ERROR: incorrect sibling of node k after removal of edge 14";
    }
    EXPECT_EQ (0, count_sibk_no14) << "ERROR: wrong number of siblings of node k after removal of edge 14";

    // Chilren of edge e10
    std::vector<int> ch10_no14 = { 15 } ;
    int count_ch10_no14  = 0 ;
    for (char_int_graph::edge_child_iterator it(e10); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (ch10_no14[count_ch10_no14++], *it) << "ERROR: incorrect child of edge 10 after removal of edge 14";
    }
    EXPECT_EQ (1, count_ch10_no14) << "ERROR: wrong number of children of edge 10 after removal of edge 14";



    // Parents of edge e18
    std::vector<int> par18_no14 = {  } ;
    int count_par18_no14  = 0 ;
    for (char_int_graph::edge_parent_iterator it(e18); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (par18_no14[count_par18_no14++], *it) << "ERROR: incorrect parent of edge 18 after removal of edge 14";
    }
    EXPECT_EQ (0, count_par18_no14) << "ERROR: wrong number of parents of edge 18 after removal of edge 14";

    // Siblings of edge e15
    std::vector<int> sib18_no14 = {  } ;
    int count_sib18_no14  = 0 ;
    for (char_int_graph::edge_sibling_iterator it(e15); it != g.edge_end(); ++it)
    {
        EXPECT_EQ (sib18_no14[count_sib18_no14++], *it) << "ERROR: incorrect sibling of edge 18 after removal of edge 14";
    }
    EXPECT_EQ (0, count_sib18_no14) << "ERROR: wrong number of siblings of edge 18 after removal of edge 14";

    /*
    digraph G {
        a -> c [ label = "1" ];
        a -> d [ label = "2" ];
        b -> d [ label = "3" ];
        b -> e [ label = "4" ];
        c -> d [ label = "5" ];
        c -> f [ label = "6" ];
        e -> h [ label = "10" ];
        f -> i [ label = "12" ];
        h -> k [ label = "15" ];
        i -> l [ label = "16" ];
        j -> i [ label = "17" ];
        j -> m [ label = "18" ];
        j -> n [ label = "19" ];
        k -> n [ label = "20" ];
        k -> o [ label = "21" ];
        l -> p [ label = "22" ];
        m -> p [ label = "23" ];
        n -> p [ label = "24" ];
        o -> p [ label = "25" ];
    }
    */

}
