#include <iostream>
#include <string>
#include "include/fathom/graph/stl_allocator.hpp"
#include "include/fathom/graph/conjoined_graph.hpp"
#include "include/fathom/graph/visualizer.hpp"

class string_1 : public std::string
{

public:

    string_1(const char* str) :
    std::string(str)
    {

    }

};

class string_2 : public std::string
{

public:

    string_2(const char* str) :
    std::string(str)
    {
        
    }

};

int main()
{
    
    mv::conjoined_graph<std::string, string_1, string_2, mv::stl_allocator> cg;

    mv::graph<std::string, string_1, mv::stl_allocator> &g1 = cg.get_first();
    mv::graph<std::string, string_2, mv::stl_allocator> &g2 = cg.get_second();

    auto g1n1It = g1.node_insert("g1_n1");
    auto g1n2It = g1.node_insert(g1n1It, "g1_n2", "g1_e1");
    auto g1n3It = g1.node_insert(g1n1It, "g1_n3", "g1_e2");
    auto g1n4It = g1.node_insert(g1n2It, "g1_n4", "g1_e3");
    auto g1n5It = g1.node_insert(g1n2It, "g1_n5", "g1_e4");


    auto g2n1It = g2.node_insert("g2_n1");
    auto g2n2It = g2.node_insert(g2n1It, "g2_n2", "g2_e1");
    auto g2n3It = g2.node_insert(g2n1It, "g2_n3", "g2_e2");


    auto g1g2n1It = g1.node_find(*g2n1It);
    auto g1g2n2It = g1.node_find(*g2n2It);
    auto g1g2n3It = g1.node_find(*g2n3It);
    auto g1e5It = g1.edge_insert(g1n1It, g1g2n1It, "g1_e5");
    auto g1e6It = g1.edge_insert(g1n3It, g1g2n2It, "g1_e6");
    auto g1e7It = g1.edge_insert(g1n5It, g1g2n3It, "g1_e7");


    auto g2g1n1It = g2.node_find(*g1n1It);
    auto g2g1n2It = g2.node_find(*g1n2It);
    auto g2g1n3It = g2.node_find(*g1n3It);
    auto g2g1n4It = g2.node_find(*g1n4It);
    auto g2g1n5It = g2.node_find(*g1n5It);

    auto g2e3It = g2.edge_insert(g2g1n1It, g2g1n2It, "g2_e3");
    auto g2e4It = g2.edge_insert(g2g1n2It, g2g1n3It, "g2_e4");
    auto g2e5It = g2.edge_insert(g2g1n3It, g2g1n4It, "g2_e5");
    auto g2e6It = g2.edge_insert(g2g1n4It, g2g1n5It, "g2_e6");
    auto g2e7It = g2.edge_insert(g2g1n5It, g2n1It, "g2_e7");

    auto itA = g1.node_begin();
    mv::conjoined_graph<char, int, bool, mv::stl_allocator>::first_graph::node_list_iterator it;

    std::cout << *itA << std::endl;

    std::cout << g1.node_size() << std::endl;
    std::cout << g2.node_size() << std::endl;

    for (auto it = g1.node_begin(); it != g1.node_end(); ++it)
        std::cout << *it << std::endl;

    for (auto it = g2.node_begin(); it != g2.node_end(); ++it)
        std::cout << *it << std::endl;

    mv::Visualizer gv(mv::node_content, mv::edge_content);
    gv.print_dot(g1);

    gv.print_dot(g2);

    g2.node_erase(g2g1n2It);

    gv.print_dot(g1);

    gv.print_dot(g2);

    g1.clear();

    std::cout << g1.node_size() << std::endl;
    std::cout << g1.edge_size() << std::endl;
    std::cout << g2.node_size() << std::endl;
    std::cout << g2.edge_size() << std::endl;

}