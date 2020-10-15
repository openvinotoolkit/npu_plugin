#include "gtest/gtest.h"
#include <exception>
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

using char_int_graph = mv::graph<char, int>;

/*
Generic sibling test:
a--->d
a--->e
a--->f
a--->g
a--->h
a--->i

b--->d
b--->e
b--->f
b--->g
b--->h
b--->i

c--->d
c--->e
c--->f
c--->g
c--->h
c--->i
 */
TEST (graph_dynamic_sibling_enumeration, sibling_test)
{
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

    auto e1 = g.edge_insert(na, nd, 1);
    auto e2 = g.edge_insert(na, ne, 2);
    auto e3 = g.edge_insert(na, nf, 3);
    auto e4 = g.edge_insert(na, ng, 4);
    auto e5 = g.edge_insert(na, nh, 5);
    auto e6 = g.edge_insert(na, ni, 6);

    auto e7 = g.edge_insert(nb, nd, 7);
    auto e8 = g.edge_insert(nb, ne, 8);
    auto e9 = g.edge_insert(nb, nf, 9);
    auto e10 = g.edge_insert(nb, ng, 10);
    auto e11 = g.edge_insert(nb, nh, 11);
    auto e12 = g.edge_insert(nb, ni, 12);

    auto e13 = g.edge_insert(nc, nd, 13);
    auto e14 = g.edge_insert(nc, ne, 14);
    auto e15 = g.edge_insert(nc, nf, 15);
    auto e16 = g.edge_insert(nc, ng, 16);
    auto e17 = g.edge_insert(nc, nh, 17);
    auto e18 = g.edge_insert(nc, ni, 18);


    // Siblings of node ne
    std::vector<char> nodes_sid={'d','e','g','h','i'};
    int count_nodes_sid= 0 ;
    for (char_int_graph::node_sibling_iterator it(nf); it != char_int_graph::node_sibling_iterator(); ++it)
    {
        EXPECT_EQ (nodes_sid[count_nodes_sid++], *it) << "ERROR: incorrect sibling of node f";
    }
    EXPECT_EQ (nf->siblings_size(), count_nodes_sid) << "ERROR: wrong number of siblings of node f";

    EXPECT_EQ ('e', *nd->leftmost_sibling()) << "ERROR: wrong leftmost sibling node of node d";
    EXPECT_EQ ('d', *nf->leftmost_sibling()) << "ERROR: wrong leftmost sibling node of node f";
    EXPECT_EQ ('d', *ni->leftmost_sibling()) << "ERROR: wrong leftmost sibling node of node i";

    EXPECT_EQ ('i', *nd->rightmost_sibling()) << "ERROR: wrong rightmost sibling node of node d";
    EXPECT_EQ ('i', *nf->rightmost_sibling()) << "ERROR: wrong rightmost sibling node of node f";
    EXPECT_EQ ('h', *ni->rightmost_sibling()) << "ERROR: wrong rightmost sibling node of node i";

}

/*
Test:
-check to get leftmost sibling: iterate over sibling of next parent
-check when sibling share only 1 parent in common

a     b
|  / | \
| /  |  \
c    d    e
 */


TEST (graph_dynamic_sibling_enumeration, sibling_test_2)
{
    char_int_graph g;

    auto na = g.node_insert('a');
    auto nb = g.node_insert('b');
    auto nc = g.node_insert('c');
    auto nd = g.node_insert('d');
    auto ne = g.node_insert('e');

    auto e1 = g.edge_insert(na, nc, 1);
    auto e2 = g.edge_insert(nb, nc, 2);
    auto e3 = g.edge_insert(nb, nd, 3);
    auto e4 = g.edge_insert(nb, ne, 4);

    EXPECT_EQ ('e', *nc->rightmost_sibling()) << "ERROR: wrong rightmost sibling node of node c";
    EXPECT_EQ ('d', *nc->leftmost_sibling()) << "ERROR: wrong leftmost sibling node of node c";

    //Test for the previous bug found while removing parent when there is only one common parent with sibling
    g.edge_erase(e1);

    // Siblings of node nc
    std::vector<char> nodes_sid={'d','e'};
    int count_nodes_sid= 0 ;
    for (char_int_graph::node_sibling_iterator it(nc); it != char_int_graph::node_sibling_iterator(); ++it)
    {
        EXPECT_EQ (nodes_sid[count_nodes_sid++], *it) << "ERROR: incorrect sibling of node c";
    }
    EXPECT_EQ (2, count_nodes_sid) << "ERROR: wrong number of siblings of node c";

}

/*
	a
   /|
  b |
   \|
    c

    sibling is also a parent
 */

TEST (graph_dynamic_sibling_enumeration, sibling_test_3)
{
    char_int_graph g;

    auto na = g.node_insert('a');
    auto nb = g.node_insert('b');
    auto nc = g.node_insert('c');

    auto e1 = g.edge_insert(na, nb, 1);
    auto e2 = g.edge_insert(na, nc, 2);
    auto e7 = g.edge_insert(nb, nc, 3);

    EXPECT_EQ ('b', *nc->leftmost_sibling()) << "ERROR: wrong leftmost sibling node of node c";
    EXPECT_EQ (nc->siblings_size(), 1) << "ERROR: wrong number of siblings of node c";

    // Siblings of node nc
    std::vector<char> nodes_sid={'b'};
    int count_nodes_sid= 0 ;
    for (char_int_graph::node_sibling_iterator it(nc); it != char_int_graph::node_sibling_iterator(); ++it)
    {
        EXPECT_EQ (nodes_sid[count_nodes_sid++], *it) << "ERROR: incorrect sibling of node c";
    }
}

/*
a  b  c
|\ | /|
| \|/ |
d  e  f
 */

TEST (graph_dynamic_sibling_enumeration, sibling_test_4)
{
    char_int_graph g;

    auto na = g.node_insert('a');
    auto nb = g.node_insert('b');
    auto nc = g.node_insert('c');
    auto nd = g.node_insert('d');
    auto ne = g.node_insert('e');
    auto nf = g.node_insert('f');

    auto e1 = g.edge_insert(na, nd, 1);
    auto e2 = g.edge_insert(na, ne, 2);
    auto e3 = g.edge_insert(nb, ne, 3);
    auto e4 = g.edge_insert(nc, ne, 4);
    auto e5 = g.edge_insert(nc, nf, 5);

    EXPECT_EQ ('e', *nf->rightmost_sibling()) << "ERROR: wrong rightmost sibling node of node f";
    EXPECT_EQ ('e', *nf->leftmost_sibling()) << "ERROR: wrong leftmost sibling node of node f";

    EXPECT_EQ (1, nf->siblings_size()) << "ERROR: wrong number of siblings of node f";
}

TEST (graph_dynamic_sibling_enumeration, sibling_test_no_sibling)
{
    char_int_graph g;

    auto na = g.node_insert('a');
    auto nb = g.node_insert('b');
    auto nc = g.node_insert('c');
    auto nd = g.node_insert('d');

    auto e1 = g.edge_insert(na, nd, 1);
    auto e2 = g.edge_insert(nb, nd, 2);
    auto e3 = g.edge_insert(nc, nd, 3);

    try
    {
     char_int_graph::node_sibling_iterator it(nd);
    }
    catch (std::exception& e)
    {
       std::cout << "Sibling iterator returns " << e.what() << std::endl;
    }
    try
    {
       auto left_sibling = *nd->leftmost_sibling();
    }
    catch (std::exception& e)
    {
       std::cout << "leftmost_sibling returns " << e.what() << std::endl;
    }
    try
    {
       auto right_Sibling = *nd->rightmost_sibling();
    }
    catch (std::exception& e)
    {
       std::cout << "rightmost_sibling returns " << e.what() << std::endl;
    }

    EXPECT_EQ (0, nd->siblings_size()) << "ERROR: wrong number of siblings of node d";
}

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

typedef mv::DAG_Transitive_Reducer<graph_string_string,
        EdgeItComparator, NodeItComparator> transitive_reducer_t;
TEST(graph_transitive_reduction, cycle) {

  graph_string_string g;

  auto na = g.node_insert("a");
  auto nb = g.node_insert("b");
  auto nc = g.node_insert("c");
  auto nd = g.node_insert("d");

  g.edge_insert(na, nb, "a->b");
  g.edge_insert(nb, nc, "b->c");
  g.edge_insert(nc, nd, "c->d");
  g.edge_insert(nd, nb, "d->b");

  transitive_reducer_t reducer(g);

  EXPECT_FALSE(reducer.reduce());
}

template<typename IteratorA, typename IteratorB>
bool EquivalentIteratorList(IteratorA abeg, IteratorA aend,
    IteratorB bbeg, IteratorB bend) {

  while (abeg != aend) {
    if (bbeg == bend) { return false; }
    if (!(*abeg == *bbeg)) {
      std::cout <<"a =" << *abeg <<" b=" << *bbeg << std::endl;
      return false;
    }
    ++abeg;
    ++bbeg;
  }
  return (abeg == aend) && (bbeg == bend);
}

bool EquivalentLabelledGraphs(graph_string_string& g_a,
    graph_string_string& g_b) {

  if (g_a.node_size() != g_b.node_size()) { return false; }
  if (g_a.edge_size() != g_b.edge_size()) { return false; }

  return EquivalentIteratorList(g_a.node_begin(), g_a.node_end(),
        g_b.node_begin(), g_b.node_end()) &&
    EquivalentIteratorList(g_a.edge_begin(), g_a.edge_end(),
        g_b.edge_begin(), g_b.edge_end());
}

typedef mv::DAG_Transitive_Reducer<graph_string_string,
        EdgeItComparator, NodeItComparator> transitive_reducer_t;

TEST(graph_transitive_reduction, simple_chain) {
  graph_string_string g, g_gold;

  {
    //input //
    auto na = g.node_insert("a");
    auto nb = g.node_insert("b");
    auto nc = g.node_insert("c");
    g.edge_insert(na, nb, "a->b");
    g.edge_insert(nb, nc, "b->c");
    g.edge_insert(na, nc, "a->c");
  }

  {
    //output//
    auto na = g_gold.node_insert("a");
    auto nb = g_gold.node_insert("b");
    auto nc = g_gold.node_insert("c");
    g_gold.edge_insert(na, nb, "a->b");
    g_gold.edge_insert(nb, nc, "b->c");
  }


  transitive_reducer_t reducer(g);
  EXPECT_TRUE(reducer.reduce());
  EXPECT_TRUE(EquivalentLabelledGraphs(g, g_gold));
}

TEST(graph_transitive_reduction, multi_level_chain) {
  graph_string_string g, g_gold;
  char node_buf[4096], edge_buf[8194];
  size_t node_count = 25UL;

  { // input: one node per level and add edges from node at
    // level 'i' to all nodes nodes at levels 'i+1', 'i+2' ... //
    auto nprev = g.node_insert("N0");
    for (size_t i=1UL; i<node_count; i++) {
      sprintf(node_buf, "N%lu", i);
      auto ncurr = g.node_insert(std::string(node_buf));

      sprintf(edge_buf, "E:%lu->%lu", i-1, i);
      g.edge_insert(nprev, ncurr, std::string(edge_buf));

      nprev = ncurr;
    }
    g_gold = g; // just a linear chain //

    for (size_t i=1UL; i<node_count; i++) {
      sprintf(node_buf, "N%lu", i);
      auto nodei = g.node_find(node_buf);

      for (size_t j=i+2UL; j<node_count; j++) {
        sprintf(node_buf, "N%lu", j);
        auto nodej = g.node_find(node_buf);

        sprintf(edge_buf, "E:%lu->%lu", i, j);
        g.edge_insert(nodei, nodej, std::string(edge_buf));
      }
    }
  }


  transitive_reducer_t reducer(g);
  EXPECT_TRUE(reducer.reduce());
  reducer.dump_reduce_info();
  EXPECT_TRUE(EquivalentLabelledGraphs(g, g_gold));
}

TEST(graph_transitive_reduction, bipartite) {
  graph_string_string g, g_gold;

  {
    //input //
    auto na = g.node_insert("a");
    auto nb = g.node_insert("b");
    auto nc = g.node_insert("c");
    auto nd = g.node_insert("d");

    g.edge_insert(na, nc, "a->c");
    g.edge_insert(na, nd, "a->d");
    g.edge_insert(nb, nc, "b->c");
    g.edge_insert(nb, nd, "b->d");
  }

  {
    g_gold = g;
  }

  transitive_reducer_t reducer(g);
  EXPECT_TRUE(reducer.reduce());
  reducer.dump_reduce_info();
  EXPECT_TRUE(EquivalentLabelledGraphs(g, g_gold));
}
