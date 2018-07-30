#include <iostream>
#include <string>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/stl_allocator.hpp"


using graph_char_int= mv::graph<char, int, mv::stl_allocator>;
using graph_char_bool = mv::graph<char, bool, mv::stl_allocator>;

int main()
{
    
    //mv::stl_allocator::callback n_callback = &foo;
    //mv::stl_allocator::alloc_fail_callback = n_callback;

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

    // Define graph
    graph_char_int g;

    //graph_char_bool g2 = graph_char_int::shallow_nodes_copy<bool>(g);

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
    std::cout << "Inputs (ng - " << ng->inputs_size() << "): ";
    for (auto it = ng->leftmost_input(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Node outputs - for node nc
    std::cout << "Outputs (nc - " << nc->outputs_size() << "): ";
    for (auto it = nc->leftmost_output(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Edge source - for edge e1
    std::cout << "Source (e1): n" << *e1->source() << std::endl;
    // Edge sink - for edge e1
    std::cout << "Sink (e1): n" << *e1->sink() << std::endl;

    // Graph nodes traversing

    // List
    std::cout << "Nodes list: ";
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Reverse list
    std::cout << "Nodes reverse list: ";
    for (graph_char_int::node_reverse_list_iterator it = g.node_rbegin(); it != g.node_rend(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Search
    auto dfs_fdir = graph_char_int::node_dfs_iterator::forward;
    auto dfs_rdir = graph_char_int::node_dfs_iterator::reverse;
    auto dfs_lside = graph_char_int::node_dfs_iterator::leftmost;
    auto dfs_rside = graph_char_int::node_dfs_iterator::rightmost;
    auto bfs_fdir = graph_char_int::node_bfs_iterator::forward;
    auto bfs_rdir = graph_char_int::node_bfs_iterator::reverse;
    auto bfs_lside = graph_char_int::node_bfs_iterator::leftmost;
    auto bfs_rside = graph_char_int::node_bfs_iterator::rightmost;

    // DFS (forward leftmost) - startng from node na
    std::cout << "Nodes DFS forward leftmost (starting na): ";
    for (graph_char_int::node_dfs_iterator it(na); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

     // DFS (forward rightmost) - startng from node na
    std::cout << "Nodes DFS forward rightmost (starting na): ";
    for (graph_char_int::node_dfs_iterator it(na, dfs_fdir, dfs_rside); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // DFS (reverse leftmost) - startng from node np
    std::cout << "Nodes DFS reverse leftmost (starting np): ";
    for (graph_char_int::node_dfs_iterator it(np, dfs_rdir, dfs_lside); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

     // DFS (reverse rigthmost)- startng from node na
    std::cout << "Nodes DFS reverse rightmost (starting np): ";
    for (graph_char_int::node_dfs_iterator it(np, dfs_rdir, dfs_rside); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // BFS (forward leftmost) - startng from node na
    std::cout << "Nodes BFS forward leftmost (starting na): ";
    for (graph_char_int::node_bfs_iterator it(na, bfs_fdir, bfs_lside); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

     // BFS (forward rightmost) - startng from node na
    std::cout << "Nodes BFS forward rightmost (starting na): ";
    for (graph_char_int::node_bfs_iterator it(na, bfs_fdir, bfs_rside); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // BFS (reverse leftmost) - startng from node np
    std::cout << "Nodes BFS reverse leftmost (starting np): ";
    for (graph_char_int::node_bfs_iterator it(np, bfs_rdir, bfs_lside); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

     // BFS (reverse rigthmost) - startng from node np
    std::cout << "Nodes BFS reverse rightmost (starting np): ";
    for (graph_char_int::node_bfs_iterator it(np, bfs_rdir, bfs_rside); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;


    // Graph edges traversing

    // List
    std::cout << "Edges list: ";
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Reverse list
    std::cout << "Edges reverse list: ";
    for (graph_char_int::edge_reverse_list_iterator it = g.edge_rbegin(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // DFS - startng from edge e1
    std::cout << "Edges DFS (starting e1): ";
    for (graph_char_int::edge_dfs_iterator it(e1); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // BFS - starting from edge e1
    std::cout << "Edges BFS (starting e1): ";
    for (graph_char_int::edge_bfs_iterator it(e1); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;


    // Relationship nodes traversing

    // Chilren of node nc
    std::cout << "Children (nc): ";
    for (graph_char_int::node_child_iterator it(nc); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Leftmost child (nc): n" << *nc->leftmost_child() << std::endl;
    std::cout << "Rightmost child (nc): n" << *nc->rightmost_child() << std::endl;

    // Parents of node np
    std::cout << "Parents (np): ";
    for (graph_char_int::node_parent_iterator it(np); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Leftmost parent (np): n" << *np->leftmost_parent() << std::endl;
    std::cout << "Rightmost parent (np): n" << *np->rightmost_parent() << std::endl;

    // Siblings of node nd
    std::cout << "Siblings (nd): ";
    for (graph_char_int::node_sibling_iterator it(nd); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Leftmost sibling (nd): n" << *nd->leftmost_sibling() << std::endl;
    std::cout << "Rightmost sibling (nd): n" << *nd->rightmost_sibling() << std::endl;

    // Relationship edges traversing

    // Chilren of edge e1
    std::cout << "Children (e1): ";
    for (graph_char_int::edge_child_iterator it(e1); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Leftmost child (e1): e" << *e1->leftmost_child() << std::endl;
    std::cout << "Rightmost child (e1): e" << *e1->rightmost_child() << std::endl;

    // Parents of edge e13
    std::cout << "Parents (e13): ";
    for (graph_char_int::edge_parent_iterator it(e13); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Leftmost parent (e13): e" << *e13->leftmost_parent() << std::endl;
    std::cout << "Rightmost parent (e13): e" << *e13->rightmost_parent() << std::endl;

    // Siblings of edge e17
    std::cout << "Siblings (e17): ";
    for (graph_char_int::edge_sibling_iterator it(e17); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    std::cout << "Leftmost sibling (e17): e" << *e17->leftmost_sibling() << std::endl;
    std::cout << "Rightmost sibling (e17): e" << *e17->rightmost_sibling() << std::endl;

    // Node removal (ng)
     
    // Initialize traversing before deletion
    graph_char_int::node_dfs_iterator dfs_nc(nc);
    graph_char_int::node_bfs_iterator bfs_nc(nc);
    graph_char_int::edge_dfs_iterator dfs_e4(e4);
    graph_char_int::edge_bfs_iterator bfs_e4(e4);

    // Delete ng
    g.node_erase(ng);

    // Check if iterators invalid
    if (ng == g.node_end())
    {
        std::cout << "Node ng removed" << std::endl;
    }
    if (e7 == g.edge_end())
    {
        std::cout << "Edge e7 removed" << std::endl;
    }
    if (e8 == g.edge_end())
    {
        std::cout << "Edge e8 removed" << std::endl;
    }
    if (e9 == g.edge_end())
    {
        std::cout << "Edge e9 removed" << std::endl;
    }
    if (e11 == g.edge_end())
    {
        std::cout << "Edge e11 removed" << std::endl;
    }
    if (e13 == g.edge_end())
    {
        std::cout << "Edge e13 removed" << std::endl;
    }

     // Nodes list
    std::cout << "Nodes list: ";
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Node reverse list
    std::cout << "Nodes reverse list: ";
    for (graph_char_int::node_reverse_list_iterator it = g.node_rbegin(); it != g.node_rend(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

     // Edges list
    std::cout << "Edges list: ";
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Edges reverse list
    std::cout << "Edges reverse list: ";
    for (graph_char_int::edge_reverse_list_iterator it = g.edge_rbegin(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // DFS - startng from node nc
    std::cout << "Nodes DFS (starting nc): ";
    for (; dfs_nc != g.node_end(); ++dfs_nc)
    {
        std::cout << "n" << *dfs_nc << " ";
    }
    std::cout << std::endl;

    // BFS - starting from node nc
    std::cout << "Nodes BFS (starting nc): ";
    for (; bfs_nc != g.node_end(); ++bfs_nc)
    {
        std::cout << "n" << *bfs_nc << " ";
    }
    std::cout << std::endl;

    // DFS - startng from edge e4
    std::cout << "Edges DFS (starting e4): ";
    for (; dfs_e4 != g.edge_end(); ++dfs_e4)
    {
        std::cout << "e" << *dfs_e4 << " ";
    }
    std::cout << std::endl;

    // BFS - starting from edge e4
    std::cout << "Edges BFS (starting e4): ";
    for (; bfs_e4 != g.edge_end(); ++bfs_e4)
    {
        std::cout << "e" << *bfs_e4 << " ";
    }
    std::cout << std::endl;

    // Chilren of node ne
    std::cout << "Children (ne): ";
    for (graph_char_int::node_child_iterator it(ne); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Parents of node nj
    std::cout << "Parents (nj): ";
    for (graph_char_int::node_parent_iterator it(nj); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Siblings of node nh
    std::cout << "Siblings (nh): ";
    for (graph_char_int::node_sibling_iterator it(nh); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Chilren of edge e5
    std::cout << "Children (e5): ";
    for (graph_char_int::edge_child_iterator it(e5); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Parents of edge e18
    std::cout << "Parents (e18): ";
    for (graph_char_int::edge_parent_iterator it(e18); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Siblings of edge e12
    std::cout << "Siblings (e12): ";
    for (graph_char_int::edge_sibling_iterator it(e12); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;


    // Edge removal (e14)
     
    // Initialize traversing before deletion
    graph_char_int::node_dfs_iterator dfs_nb(nb);
    graph_char_int::node_bfs_iterator bfs_nb(nb);
    dfs_e4 = e4;    
    bfs_e4 = e4;

    // Delete e14
    g.edge_erase(e14);

    // Check if iterator invalid
    if (e14 == g.edge_end())
    {
        std::cout << "Edge e14 removed" << std::endl;
    }

     // Nodes list
    std::cout << "Nodes list: ";
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Node reverse list
    std::cout << "Nodes reverse list: ";
    for (graph_char_int::node_reverse_list_iterator it = g.node_rbegin(); it != g.node_rend(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

     // Edges list
    std::cout << "Edges list: ";
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Edges reverse list
    std::cout << "Edges reverse list: ";
    for (graph_char_int::edge_reverse_list_iterator it = g.edge_rbegin(); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // DFS - startng from node nb
    std::cout << "Nodes DFS (starting nb): ";
    for (; dfs_nb != g.node_end(); ++dfs_nb)
    {
        std::cout << "n" << *dfs_nb << " ";
    }
    std::cout << std::endl;

    // BFS - starting from node nb
    std::cout << "Nodes BFS (starting nb): ";
    for (; bfs_nb != g.node_end(); ++bfs_nb)
    {
        std::cout << "n" << *bfs_nb << " ";
    }
    std::cout << std::endl;

    // DFS - startng from edge e4
    std::cout << "Edges DFS (starting e4): ";
    for (; dfs_e4 != g.edge_end(); ++dfs_e4)
    {
        std::cout << "e" << *dfs_e4 << " ";
    }
    std::cout << std::endl;

    // BFS - starting from edge e4
    std::cout << "Edges BFS (starting e4): ";
    for (; bfs_e4 != g.edge_end(); ++bfs_e4)
    {
        std::cout << "e" << *bfs_e4 << " ";
    }
    std::cout << std::endl;

    // Chilren of node nh
    std::cout << "Children (nh): ";
    for (graph_char_int::node_child_iterator it(nh); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Parents of node nj
    std::cout << "Parents (nj): ";
    for (graph_char_int::node_parent_iterator it(nj); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Siblings of node nk
    std::cout << "Siblings (nk): ";
    for (graph_char_int::node_sibling_iterator it(nk); it != g.node_end(); ++it)
    {
        std::cout << "n" << *it << " ";
    }
    std::cout << std::endl;

    // Chilren of edge e10
    std::cout << "Children (e10): ";
    for (graph_char_int::edge_child_iterator it(e10); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Parents of edge e18
    std::cout << "Parents (e18): ";
    for (graph_char_int::edge_parent_iterator it(e18); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

    // Siblings of edge e15
    std::cout << "Siblings (e15): ";
    for (graph_char_int::edge_sibling_iterator it(e15); it != g.edge_end(); ++it)
    {
        std::cout << "e" << *it << " ";
    }
    std::cout << std::endl;

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

    return 0;

}