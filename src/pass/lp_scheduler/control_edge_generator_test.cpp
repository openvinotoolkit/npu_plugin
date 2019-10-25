
#include <gtest/gtest.h>
#include <vector>

#include "pass/lp_scheduler/control_edge_generator.hpp"
#include "scheduler/scheduler_unit_test_utils.hpp"

typedef mv::pass::Control_Edge_Generator<mv_unit_tests::interval_t>
  control_edge_generator_t;

class Control_Edge_Generator_Test
  : public ::testing::Test, control_edge_generator_t {

  protected:

  typedef control_edge_generator_t::interval_t interval_t;
  typedef control_edge_generator_t::interval_tree_t interval_tree_t;
  typedef control_edge_generator_t::query_iterator_t query_iterator_t;
  typedef control_edge_generator_t::noop_functor_t noop_functor_t;

  void SetUp() {}

  const interval_tree_t& interval_tree() const { return interval_tree_; }
  void process_next_interval(const interval_t& interval) {
    noop_functor_t noop;
    control_edge_generator_t::process_next_interval(interval, noop);
  }


  template<typename IntervalIterator>
  bool is_disjoint_interval_set_equivalent_to(IntervalIterator ibegin,
      IntervalIterator iend) {

    query_iterator_t qitr = interval_tree_.begin(),
      qitr_end = interval_tree_.end();

    for (;(qitr != qitr_end) && (ibegin != iend); ++qitr, ++ibegin) {
      interval_t qinterval(qitr.interval_begin(), qitr.interval_end(),
          (*qitr).id_);

      if (!(qinterval == *ibegin)) { return false; }
    }
    return (qitr == qitr_end) && (ibegin == iend);
  }

  void dump() {
    query_iterator_t qitr = interval_tree_.begin(),
      qitr_end = interval_tree_.end();

    for (; qitr != qitr_end; ++qitr) {
      std::cout << " [" << qitr.interval_begin() << " " << qitr.interval_end()
        << " " << (*qitr).id_ << "]" << std::endl;
    }
  }


}; // class Control_Edge_Generator_Test //

TEST_F(Control_Edge_Generator_Test, interval_tree_state) {
  {
    std::vector<interval_t> expected = { {1, 10, "a"} };

    process_next_interval(interval_t(1,10, "a"));
    EXPECT_TRUE(is_disjoint_interval_set_equivalent_to(expected.begin(),
          expected.end()));
  }

  { 
    // fully contained //
    std::vector<interval_t> expected = { {1,3,"a"}, {4,8,"b"}, {9,10,"a"} };
    process_next_interval(interval_t(4,8, "b"));
    EXPECT_TRUE(is_disjoint_interval_set_equivalent_to(expected.begin(),
          expected.end()));
  }

  {
    //add a disjoint interval. //
    std::vector<interval_t> expected = { {1,3,"a"}, {4,8,"b"}, {9,10,"a"}, 
      {15, 20, "f"} };
    process_next_interval(interval_t(15 ,20, "f"));
    EXPECT_TRUE(is_disjoint_interval_set_equivalent_to(expected.begin(),
          expected.end()));
  }

  {
    // "c" covers all the existing intervals and removes everyone. //
    std::vector<interval_t> expected = { {0,12,"c"}, {15,20,"f"} };
    process_next_interval(interval_t(0,12, "c"));
    EXPECT_TRUE(is_disjoint_interval_set_equivalent_to(expected.begin(),
          expected.end()));
  }

  {
    // "e" covers both "c" and "f" partially. //
    std::vector<interval_t> expected = { {0,7,"c"}, {8,16,"e"}, {17,20,"f"} };
    process_next_interval(interval_t(8,16, "e"));
    EXPECT_TRUE(is_disjoint_interval_set_equivalent_to(expected.begin(),
          expected.end()));
  }

  {
    // "h" touches end points //
    std::vector<interval_t> expected { {0,6,"c"}, {7,17,"h"}, {18,20,"f"} };
    process_next_interval(interval_t(7,17, "h"));
    EXPECT_TRUE(is_disjoint_interval_set_equivalent_to(expected.begin(),
          expected.end()));
  }
}

TEST_F(Control_Edge_Generator_Test, extract_control_edges) {
  //
  //   +------+-----+
  //   |      |  B  +--------+
  //   |      +-----+        +--------+
  //   |  A   |#####|   D    |        |
  //   |      +-----+--------+   F    |
  //   |      |       C      |        |
  //   +------+--------------+--------+
  //

  std::vector<interval_t> packing = {
    {0,10,"A"}, // time=0 //
    {0,3,"C"}, {7,10,"B"}, // time=1 //
    {4,9,"D"}, //time=2//
    {0,6,"F"} //time=3//
  };

  struct control_edges_t {
   void operator()(const interval_t& a, const interval_t& b) {
     if (a.id_ < b.id_) {
       edges_.insert(a.id_+","+b.id_);
     } else {
       edges_.insert(b.id_+","+a.id_);
     }
   }
   std::unordered_set<std::string> edges_;
  };

  control_edge_generator_t algo;
  control_edges_t edges;

  size_t edge_count = algo.generate_control_edges(
      packing.begin(), packing.end(), edges);

  EXPECT_EQ(edge_count, 6UL);

  std::unordered_set<std::string> expected_edges = {
    {"A,B"}, {"A,C"}, {"A,D"}, {"B,D"}, {"C,F"}, {"D,F"} };
  EXPECT_EQ(expected_edges, edges.edges_);
}
