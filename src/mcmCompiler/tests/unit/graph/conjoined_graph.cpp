#include "gtest/gtest.h"
#include "include/mcm/graph/conjoined_graph.hpp"
#include "include/mcm/graph/visualizer.hpp"
#include <string>

class Str1 : public std::string
{

public:
    
    Str1(const char* str) : std::string(str)
    {
        
    }

};

class Str2 : public std::string
{

public:
    
    Str2(const char* str) : std::string(str)
    {

    }

};

using str_conjoined_graph = mv::conjoined_graph<std::string, Str1, Str2>;


TEST(conjoined_graph, skip_child)
{

    str_conjoined_graph g;
    str_conjoined_graph::first_graph& g1 = g.get_first();
    str_conjoined_graph::second_graph& g2 = g.get_second();

    auto g1n0 = g1.node_insert("n0");
    auto g2n0 = g.get_second_iterator(g1n0);
    auto g1n1 = g1.node_insert(g1n0, "n1", "g1e0");
    auto g1e0 = g1n0->leftmost_output();
    auto g2n1 = g.get_second_iterator(g1n1);
    auto g2e0 = g2.edge_insert(g2n0, g2n1, "g2e0");
    auto g1n2 = g1.node_insert(g1n1, "n2", "g1e1");
    auto g1e1 = g1n1->leftmost_output();
    auto g2n2 = g.get_second_iterator(g1n2);
    auto g2e1 = g2.edge_insert(g2n1, g2n2, "g2e1");
    auto g1n3 = g1.node_insert(g1n2, "n3", "g1e2");
    auto g1e2 = g1n2->leftmost_output();
    auto g2n3 = g.get_second_iterator(g1n3);
    auto g2e2 = g2.edge_insert(g2n2, g2n3, "g2e2");

    ASSERT_EQ(g1.node_size(), 4);
    ASSERT_EQ(g2.node_size(), 4);

    ASSERT_EQ(g1n0->children_size(), 1);
    ASSERT_EQ(g1n0->parents_size(), 0);
    ASSERT_EQ(g1n0->siblings_size(), 0);
    ASSERT_EQ(g1n1->children_size(), 1);
    ASSERT_EQ(g1n1->parents_size(), 1);
    ASSERT_EQ(g1n1->siblings_size(), 0);
    ASSERT_EQ(g1n2->children_size(), 1);
    ASSERT_EQ(g1n2->parents_size(), 1);
    ASSERT_EQ(g1n2->siblings_size(), 0);
    ASSERT_EQ(g1n3->children_size(), 0);
    ASSERT_EQ(g1n3->parents_size(), 1);
    ASSERT_EQ(g1n3->siblings_size(), 0);

    auto g1It = g1.node_begin();
    for (auto g2It = g2.node_begin(); g2It != g2.node_end(); ++g2It)
    {
        ASSERT_EQ(*g1It, *g2It);
        ASSERT_EQ(g1It->children_size(), g2It->children_size());
        ASSERT_EQ(g1It->parents_size(), g2It->parents_size());
        ASSERT_EQ(g1It->siblings_size(), g2It->siblings_size());
        ++g1It;
    }

    ASSERT_EQ(g1.edge_size(), 3);
    ASSERT_EQ(g2.edge_size(), 3);

    ASSERT_EQ(g1e0->children_size(), 1);
    ASSERT_EQ(g1e0->parents_size(), 0);
    ASSERT_EQ(g1e0->siblings_size(), 0);
    ASSERT_EQ(g1e1->children_size(), 1);
    ASSERT_EQ(g1e1->parents_size(), 1);
    ASSERT_EQ(g1e1->siblings_size(), 0);
    ASSERT_EQ(g1e2->children_size(), 0);
    ASSERT_EQ(g1e2->parents_size(), 1);
    ASSERT_EQ(g1e2->siblings_size(), 0);

    ASSERT_EQ(g2e0->children_size(), 1);
    ASSERT_EQ(g2e0->parents_size(), 0);
    ASSERT_EQ(g2e0->siblings_size(), 0);
    ASSERT_EQ(g2e1->children_size(), 1);
    ASSERT_EQ(g2e1->parents_size(), 1);
    ASSERT_EQ(g2e1->siblings_size(), 0);
    ASSERT_EQ(g2e2->children_size(), 0);
    ASSERT_EQ(g2e2->parents_size(), 1);
    ASSERT_EQ(g2e2->siblings_size(), 0);

    auto g1n4 = g1.node_insert("n4");
    auto g2n4 = g.get_second_iterator(g1n4);
    auto g1e3 = g1.edge_insert(g1n1, g1n4, "g1e3");
    auto g1e4 = g1.edge_insert(g1n4, g1n3, "g1e4");
    auto g2e3 = g2.edge_insert(g2n1, g2n4, "g2e3");
    auto g2e4 = g2.edge_insert(g2n4, g2n3, "g2e4");
    g1.node_erase(g1n2);
    auto g1e5 = g1.edge_insert(g1n1, g1n3, "g1e5");
    auto g2e5 = g2.edge_insert(g2n1, g2n3, "g2e5");
    g1.node_erase(g1n4);

    ASSERT_EQ(g1.node_size(), 3);
    ASSERT_EQ(g2.node_size(), 3);

    ASSERT_TRUE(g1n2 == g1.node_end());
    ASSERT_TRUE(g1n4 == g1.node_end());

    ASSERT_TRUE(g2n2 == g2.node_end());
    ASSERT_TRUE(g2n4 == g2.node_end());

    ASSERT_EQ(g1n0->children_size(), 1);
    ASSERT_EQ(g1n0->parents_size(), 0);
    ASSERT_EQ(g1n0->siblings_size(), 0);
    ASSERT_EQ(g1n1->children_size(), 1);
    ASSERT_EQ(g1n1->parents_size(), 1);
    ASSERT_EQ(g1n1->siblings_size(), 0);
    ASSERT_EQ(g1n3->children_size(), 0);
    ASSERT_EQ(g1n3->parents_size(), 1);
    ASSERT_EQ(g1n3->siblings_size(), 0);

    g1It = g1.node_begin();
    for (auto g2It = g2.node_begin(); g2It != g2.node_end(); ++g2It)
    {
        ASSERT_EQ(*g1It, *g2It);
        ASSERT_EQ(g1It->children_size(), g2It->children_size());
        ASSERT_EQ(g1It->parents_size(), g2It->parents_size());
        ASSERT_EQ(g1It->siblings_size(), g2It->siblings_size());
        ++g1It;
    }

    ASSERT_EQ(g1.edge_size(), 2);
    ASSERT_EQ(g2.edge_size(), 2);

    ASSERT_TRUE(g1e1 == g1.edge_end());
    ASSERT_TRUE(g1e2 == g1.edge_end());
    ASSERT_TRUE(g1e3 == g1.edge_end());
    ASSERT_TRUE(g1e4 == g1.edge_end());
    
    ASSERT_TRUE(g2e1 == g2.edge_end());
    ASSERT_TRUE(g2e2 == g2.edge_end());
    ASSERT_TRUE(g2e3 == g2.edge_end());
    ASSERT_TRUE(g2e4 == g2.edge_end());

    ASSERT_EQ(g1e0->children_size(), 1);
    ASSERT_EQ(g1e0->parents_size(), 0);
    ASSERT_EQ(g1e0->siblings_size(), 0);
    ASSERT_EQ(g1e5->children_size(), 0);
    ASSERT_EQ(g1e5->parents_size(), 1);
    ASSERT_EQ(g1e5->siblings_size(), 0);

    ASSERT_EQ(g2e0->children_size(), 1);
    ASSERT_EQ(g2e0->parents_size(), 0);
    ASSERT_EQ(g2e0->siblings_size(), 0);
    ASSERT_EQ(g2e5->children_size(), 0);
    ASSERT_EQ(g2e5->parents_size(), 1);
    ASSERT_EQ(g2e5->siblings_size(), 0);

}