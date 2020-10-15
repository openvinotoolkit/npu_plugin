
#include <gtest/gtest.h>
#include <iterator>
#include <vector>

#include "pass/lp_scheduler/control_edge_generator.hpp"
#include "scheduler/scheduler_unit_test_utils.hpp"

typedef mv::lp_scheduler::Control_Edge_Generator<mv_unit_tests::interval_t>
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
  size_t process_next_interval(const interval_t& interval) {
    noop_functor_t noop;
    return control_edge_generator_t::process_next_interval(interval, noop);
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

TEST_F(Control_Edge_Generator_Test, real_state_from_yolo) {

  size_t ecount = 100UL;
  {
    std::vector<interval_t> expected = { {1, 43264, "a"}, {43265, 129792, "b"},
      {129793, 216320, "c"}, {216321, 432640, "d"} };

      //op=model/max_pooling2d_4/MaxPool#94_spilledWrite ddr=[1 86528]          
    ecount = process_next_interval(interval_t(1,43264, "a"));
    EXPECT_EQ(ecount, 0UL);

    ecount = process_next_interval(interval_t(43265,129792, "b"));
    EXPECT_EQ(ecount, 0UL);

    ecount = process_next_interval(interval_t(129793, 216320, "c"));
    EXPECT_EQ(ecount, 0UL);

    ecount = process_next_interval(interval_t(216321, 432640, "d"));
    EXPECT_EQ(ecount, 0UL);

    EXPECT_TRUE(is_disjoint_interval_set_equivalent_to(expected.begin(),
          expected.end()));
  }

  {
    std::vector<interval_t> expected = { {1, 86528, "e"}, {86529, 129792, "b"},
      {129793, 216320, "c"}, {216321, 432640, "d"} };
    ecount = process_next_interval(interval_t(1, 86528, "e"));
    EXPECT_EQ(ecount, 2UL);
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


class Memory_Control_Edge_Fixture : public ::testing::Test {

  protected:
    ////////////////////////////////////////////////////////////////////////////
    typedef scheduler_unit_tests::Operation_Dag dag_t;
    typedef mv::lp_scheduler::scheduler_traits<dag_t> traits;
    typedef typename traits::operation_t operation_t;
    typedef typename traits::scheduled_op_t scheduled_op_t;
    typedef typename traits::schedule_time_t schedule_time_t;
    typedef typename traits::resource_t resource_t;

    struct select_all_ops_t {
      bool operator()(const dag_t&, const operation_t&) { return true; }
    }; // struct select_all_ops_t //

    typedef mv::lp_scheduler::Memory_Control_Edge_Generator<dag_t,
            select_all_ops_t> memory_control_edge_generator_t;
    typedef typename memory_control_edge_generator_t::memory_control_edge_t
        memory_control_edge_t;
    struct memory_control_edge_ordering_t {
      bool operator()(const memory_control_edge_t& a,
            const memory_control_edge_t& b) const {
        return (a.source_ == b.source_) ? 
          (a.sink_ < b.sink_) : (a.source_ < b.source_);
      }
    }; // memory_control_edge_ordering_t //
    typedef std::set<memory_control_edge_t, memory_control_edge_ordering_t>
        control_edge_set_t;
    ////////////////////////////////////////////////////////////////////////////
    
    void SetUp() {}

    void add_scheduled_op(const operation_t& op, schedule_time_t t, 
        resource_t rbegin, resource_t rend) {
      scheduled_ops.push_back( scheduled_op_t(op, t, rbegin, rend) );
    }
    void add_scheduled_op_with_no_resources(
          const operation_t& op, schedule_time_t t) {
      resource_t rbegin = std::numeric_limits<resource_t>::max();
      resource_t rend = std::numeric_limits<resource_t>::min();

      scheduled_ops.push_back( scheduled_op_t(op, t, rbegin, rend) );
    }

    void run_generator(const dag_t& in) {
      result.clear();
      generator.generate(in, scheduled_ops.begin(), scheduled_ops.end(),
      std::inserter(result, result.end()));
    }

    bool result_has_edge(const operation_t& a, const operation_t& b) const {
      memory_control_edge_t mkey;
      mkey.source_ = a; mkey.sink_ = b;

      return result.find(mkey) != result.end();
    }


    memory_control_edge_generator_t generator;
    std::vector<scheduled_op_t> scheduled_ops;
    control_edge_set_t result;
}; // class Memory_Control_Edge_Fixture //


TEST_F(Memory_Control_Edge_Fixture, no_memory_control_edges) {
  //  +----+
  //  | A  |
  //  +----+
  //       +-----+
  //       |  B  |
  //       +-----+
  //             +-----+
  //             |  C  |
  //             +-----+
  dag_t::adjacency_map_t in = { {"start", {"A", "B", "C"}}, {"A", {}},
                                {"B", {}}, {"C", {}} };
  dag_t dag(in);

  // create schedule input //
  add_scheduled_op_with_no_resources("start", 0UL);
  add_scheduled_op("A", 1UL, 15UL, 20UL); // (op, time, start, end) //
  add_scheduled_op("B", 2UL, 10UL, 14UL);
  add_scheduled_op("C", 3UL, 1UL, 9UL);

  run_generator(in);
  EXPECT_TRUE(result.empty());
}


TEST_F(Memory_Control_Edge_Fixture, typical_gap) {
  //  +----+
  //  |    |
  //  |    | 
  //  |    +-----+
  //  | A  |  B  |
  //  |    +-----+
  //  |    |     +-----+
  //  |    |     |  C  |
  //  +----+     +-----+
  dag_t::adjacency_map_t in = { {"start", {"A", "B", "C"}}, {"A", {}},
                                {"B", {}}, {"C", {}} };
  dag_t dag(in);

  // create schedule input //
  add_scheduled_op_with_no_resources("start", 0UL);
  add_scheduled_op("A", 1UL, 1UL, 20UL); // (op, time, start, end) //
  add_scheduled_op("B", 2UL, 10UL, 14UL);
  add_scheduled_op("C", 3UL, 1UL, 9UL);

  run_generator(in);
  ASSERT_FALSE(result.empty());
  EXPECT_EQ(result.size(), 2UL);
  EXPECT_TRUE(result_has_edge("A", "B"));
  EXPECT_FALSE(result_has_edge("B", "A"));

  EXPECT_TRUE(result_has_edge("A", "C"));
  EXPECT_FALSE(result_has_edge("C", "A"));

  EXPECT_FALSE(result_has_edge("B", "C"));
}

TEST_F(Memory_Control_Edge_Fixture, force_control_consumer_edge) {
  //  +----+
  //  |    |
  //  |    | 
  //  |    +-----+
  //  | A  |  B  |
  //  |    +-----+
  //  |    |     +-----+
  //  |    |     |  C  |
  //  +----+     +-----+
  dag_t::adjacency_map_t in = { {"start", {"A", "B", "C"}}, {"A", {"B"}},
                                {"B", {}}, {"C", {}} };
  dag_t dag(in);

  // create schedule input //
  add_scheduled_op_with_no_resources("start", 0UL);
  add_scheduled_op("A", 1UL, 1UL, 20UL); // (op, time, start, end) //
  add_scheduled_op("B", 2UL, 10UL, 14UL);
  add_scheduled_op("C", 3UL, 1UL, 9UL);

  run_generator(in);
  ASSERT_FALSE(result.empty());
  EXPECT_EQ(result.size(), 3UL);
  EXPECT_TRUE(result_has_edge("A", "B"));
  EXPECT_FALSE(result_has_edge("B", "A"));

  EXPECT_TRUE(result_has_edge("A", "C"));
  EXPECT_FALSE(result_has_edge("C", "A"));

  EXPECT_TRUE(result_has_edge("B", "C"));
  EXPECT_FALSE(result_has_edge("C", "B"));
}

TEST_F(Memory_Control_Edge_Fixture, force_control_consumer_edge2) {
  //  +----+     +-----+ 
  //  |    |     |     |
  //  |    |     |  D  |
  //  |    +-----+-----+
  //  | A  |  B  |
  //  |    +-----+
  //  |    |     +-----+
  //  |    |     |  C  |
  //  +----+     +-----+
  dag_t::adjacency_map_t in = { {"start", {"A", "B", "C"}}, {"A", {"B", "D"}},
                                {"B", {}}, {"C", {}}, {"D", {}} };
  dag_t dag(in);

  // create schedule input //
  add_scheduled_op_with_no_resources("start", 0UL);
  add_scheduled_op("A", 1UL, 1UL, 20UL); // (op, time, start, end) //
  add_scheduled_op("B", 2UL, 10UL, 14UL);
  add_scheduled_op("C", 3UL, 1UL, 9UL);
  add_scheduled_op("D", 3UL, 15UL, 20UL);

  run_generator(in);
  ASSERT_FALSE(result.empty());
  EXPECT_EQ(result.size(), 5UL);
  EXPECT_TRUE(result_has_edge("A", "B"));
  EXPECT_FALSE(result_has_edge("B", "A"));

  EXPECT_TRUE(result_has_edge("A", "C"));
  EXPECT_FALSE(result_has_edge("C", "A"));

  EXPECT_TRUE(result_has_edge("A", "D"));
  EXPECT_FALSE(result_has_edge("D", "A"));

  // this will be dropped in transitive reduction //
  EXPECT_TRUE(result_has_edge("B", "D"));
  EXPECT_FALSE(result_has_edge("D", "B"));

  EXPECT_TRUE(result_has_edge("B", "C"));
  EXPECT_FALSE(result_has_edge("C", "B"));
}
