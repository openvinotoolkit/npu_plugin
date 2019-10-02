#include <limits>
#include "gtest/gtest.h"
#include "feasible_scheduler.hpp"

namespace scheduler_unit_tests {

// Simple DAG for unit testing the core algorithm //
class Operation_Dag {
  public:
    typedef Operation_Dag dag_t;
    typedef std::string operation_t;
    typedef std::vector<operation_t> adjacency_list_t;
    typedef typename adjacency_list_t::const_iterator const_adj_list_iterator_t;
    typedef std::map<operation_t, adjacency_list_t> adjacency_map_t;
    typedef typename adjacency_map_t::const_iterator const_adj_map_iterator_t;

    class const_operation_iterator_t {
      public:
        const_operation_iterator_t() : adj_map_ptr_(NULL),
          itr_begin_(), itr_end_(), itr_adj_begin_(), itr_adj_end_(),
          iterate_edges_(false) {}

        const_operation_iterator_t(const adjacency_map_t& adj_map,
            const_adj_map_iterator_t begin, const_adj_map_iterator_t end,
            bool iterate_edges=false) : adj_map_ptr_(&adj_map), itr_begin_(begin),
          itr_end_(end), itr_adj_begin_(), itr_adj_end_(),
          iterate_edges_(iterate_edges)
        {

          if (iterate_edges_ && (itr_begin_ != itr_end_)) {
            itr_adj_begin_ = (itr_begin_->second).cbegin();
            itr_adj_end_ = (itr_begin_->second).cend();
            if (itr_adj_begin_ == itr_adj_end_) {
              move_to_next_edge();
            }
          }
        }

        // only invalid iterators are equivalent //
        bool operator==(const const_operation_iterator_t& o) const {
          return !is_valid() && !o.is_valid();
        }

        bool operator!=(const const_operation_iterator_t& o) const {
          return !(*this == o);
        }

        const const_operation_iterator_t& operator++() {
          move_to_next_vertex_or_edge();
          return *this;
        }

        const const_operation_iterator_t& operator=(
            const const_operation_iterator_t& o) {
          if (this != &o) {
            adj_map_ptr_ = o.adj_map_ptr_;
            itr_begin_ = o.itr_begin_;
            itr_end_ = o.itr_end_;
            itr_adj_begin_ = o.itr_adj_begin_;
            itr_adj_end_ = o.itr_adj_end_;
            iterate_edges_ = o.iterate_edges_;
          }
          return *this;
        }

        const operation_t& operator*() const {
          const_adj_map_iterator_t op_itr = itr_begin_;
          if (iterate_edges_) {
            op_itr = adj_map_ptr_->find(*itr_adj_begin_);
            assert(op_itr != adj_map_ptr_->end());
          }
          return op_itr->first;
        }


      private:

        // Precondition: is_valid() //
        bool move_to_next_vertex_or_edge() {
          return iterate_edges_ ? move_to_next_edge() :
              (++itr_begin_) == itr_end_;
        }

        bool move_to_next_edge() {

          if (itr_adj_begin_ != itr_adj_end_) {
            // move to edge if possible //
            ++itr_adj_begin_;
          }

          if (itr_adj_begin_ == itr_adj_end_) {
            ++itr_begin_; 
            // find a node with atleast one out going edge //
            while ( (itr_begin_ != itr_end_) &&
                    ( (itr_adj_begin_ = (itr_begin_->second).begin()) ==
                      (itr_adj_end_ = (itr_begin_->second).end()) ) ) {
              ++itr_begin_;
            }
          }

          return (itr_adj_begin_ != itr_adj_end_);
        }

        bool is_valid(void) const { return itr_begin_ != itr_end_; }

        adjacency_map_t const * adj_map_ptr_;
        const_adj_map_iterator_t itr_begin_;
        const_adj_map_iterator_t itr_end_;
        const_adj_list_iterator_t itr_adj_begin_;
        const_adj_list_iterator_t itr_adj_end_;
        bool iterate_edges_;
    }; // class const_operation_iterator_t //


    Operation_Dag(const adjacency_map_t& in) : adj_map_(in) {} 
    Operation_Dag() : adj_map_() {} 


    const_operation_iterator_t begin(const operation_t& op) const {
      adjacency_map_t::const_iterator itr = adj_map_.find(op), itr_next;
      if (itr == adj_map_.end()) { return const_operation_iterator_t(); }

      itr_next = itr;
      ++itr_next;
      return const_operation_iterator_t(adj_map_, itr, itr_next, true);
    }

    const_operation_iterator_t end(const operation_t& op) const {
      (void) op;
      return const_operation_iterator_t();
    }

    const_operation_iterator_t begin() const { 
      return const_operation_iterator_t(adj_map_, adj_map_.begin(),
          adj_map_.end(), false);
    }

    const_operation_iterator_t begin_edges() const {
      return const_operation_iterator_t(adj_map_, adj_map_.begin(),
          adj_map_.end(), true);
    }

    const_operation_iterator_t end() const {
      return const_operation_iterator_t();
    }


    // scheduler_traits //
    static const_operation_iterator_t operations_begin(const dag_t& g) {
      return g.begin();
    }

    static const_operation_iterator_t operations_end(const dag_t& g) {
      return g.end();
    }

    static const_operation_iterator_t outgoing_operations_begin(const dag_t& g,
        const operation_t& op) { return g.begin(op); }

    static const_operation_iterator_t outgoing_operations_end(const dag_t& g,
        const operation_t& op) { return g.end(op); }

  private:
    adjacency_map_t adj_map_;
}; // class Operation_Dag //

// Defines a resource model and also keeps track of the resource state //
class Resource_Model_State {
  public:
    typedef size_t resource_t;
    typedef Resource_Model_State resource_state_t;

    Resource_Model_State(
        size_t resource_bound=std::numeric_limits<resource_t>::max())
      : resources_in_use_(0UL), resource_bound_(resource_bound) {}

  private:
    size_t resources_in_use_;
    const size_t resource_bound_;
}; // class Resource_Model_State //


struct Unit_Delay_Model {
  typedef size_t delay_t;
  static delay_t delay(const Operation_Dag::operation_t& op) {
    (void)op;
    return 1;
  }
}; // struct Delay_Model //

} // namespace scheduler_unit_tests //

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits<scheduler_unit_tests::Operation_Dag>
  : public scheduler_unit_tests::Operation_Dag,
    public scheduler_unit_tests::Resource_Model_State,
    public scheduler_unit_tests::Unit_Delay_Model {

    using scheduler_unit_tests::Operation_Dag::Operation_Dag;
    using scheduler_unit_tests::Resource_Model_State::Resource_Model_State;
    using scheduler_unit_tests::Unit_Delay_Model::Unit_Delay_Model;

}; // specialization for scheduler_unit_tests::dag_t //

} // namespace lp_scheduler //
} // namespace mv //

using namespace scheduler_unit_tests;
typedef Operation_Dag dag_t;

// Unit tests to verify the iterator in the dag //
TEST(scheduler_unit_test_infra, empty_graph) {
  dag_t::adjacency_map_t in = { }; 
  dag_t dag(in);

  EXPECT_TRUE(dag.begin() == dag.end());
  EXPECT_TRUE(dag.begin_edges() == dag.end());
}

TEST(scheduler_unit_test_infra, edge_iteration_no_edges) {
  dag_t::adjacency_map_t in = { {"u", {}  }, {"v", {}  }, {"w", {} }};
  dag_t dag(in);

  dag_t::const_operation_iterator_t itr = dag.begin_edges(),
      itr_end = dag.end();

  EXPECT_TRUE(itr == itr_end);
}

TEST(scheduler_unit_test_infra, edge_iteration_with_edges) {
  dag_t::adjacency_map_t in = { {"u", {}  }, {"v", {"p", "q"}  }, {"w", {} }, 
    {"p", {}}, {"q", {}} };
  dag_t dag(in);

  dag_t::const_operation_iterator_t itr = dag.begin_edges(),
      itr_end = dag.end();
  EXPECT_FALSE(itr == itr_end);

  dag_t::adjacency_list_t expected = {"p", "q" };
  dag_t::adjacency_list_t out;
  size_t itr_count = 0;

  while (itr != itr_end) {
    ASSERT_TRUE(itr_count < 2UL);
    out.push_back(*itr);
    ++itr;
    ++itr_count;
  }

  EXPECT_EQ(expected, out);
}

TEST(scheduler_unit_test_infra, vertex_iteration_no_edges) {
  dag_t::adjacency_map_t in = { {"u", {}  }, {"v", {}  }, {"w", {} }};
  dag_t dag(in);

  dag_t::const_operation_iterator_t itr = dag.begin(), itr_end = dag.end();

  ASSERT_FALSE(itr == itr_end);
  
  dag_t::adjacency_map_t out;
  size_t itr_count = 0UL;

  while (itr != itr_end) {
    ASSERT_TRUE(itr_count < in.size());
    ASSERT_TRUE(out.find(*itr) == out.end());

    out.insert(std::make_pair(*itr, dag_t::adjacency_list_t()));
    ++itr_count;
    ++itr;
  }
  EXPECT_EQ(in, out);
}

TEST(scheduler_unit_test_infra, vertex_iteration_with_edges) {
  dag_t::adjacency_map_t in = { {"u", {"e", "f", "g"}  },
      {"v", {"h", "i", "j"}  }, {"w", {"k"} }};
  dag_t dag(in);

  dag_t::const_operation_iterator_t itr = dag.begin(), itr_end = dag.end();

  ASSERT_FALSE(itr == itr_end);
 
  dag_t::adjacency_map_t expected = { {"u", {}}, {"v", {}}, {"w", {}} };
  dag_t::adjacency_map_t out;
  size_t itr_count = 0UL;

  while (itr != itr_end) {
    ASSERT_TRUE(itr_count < in.size());
    ASSERT_TRUE(out.find(*itr) == out.end());

    out.insert(std::make_pair(*itr, dag_t::adjacency_list_t()));
    ++itr_count;
    ++itr;
  }
  EXPECT_EQ(expected, out);
}

TEST(scheduler_unit_test_infra, outgoing_edges_of_node) {
  dag_t::adjacency_map_t in = { {"u", {"v"}  }, {"v", {"w"}  }, {"w", {"x"} },
      {"x", {} } }; // chain //
  dag_t dag(in);

  dag_t::const_operation_iterator_t itr = dag.begin("u"),
      itr_end = dag.end("u");
  EXPECT_FALSE(itr == itr_end);
  EXPECT_TRUE(itr != itr_end);
  EXPECT_EQ(*itr, "v");
  ++itr;
  EXPECT_TRUE(itr == itr_end);
}

typedef mv::lp_scheduler::Feasible_Schedule_Generator<Operation_Dag>
  scheduler_t; 

// Unit tests for the scheduler algorithm //
TEST(scheduler_basic_test, chain_scheduler) {
  dag_t::adjacency_map_t in = { {"u", {"v"}  }, {"v", {"w"}  }, {"w", {"x"} },
      {"x", {} } }; // chain //
  dag_t g(in);
  scheduler_t scheduler(g);

  EXPECT_EQ(scheduler.current_time(), 0UL);
  EXPECT_TRUE( scheduler.begin_candidates() != scheduler.end_candidates() );

  scheduler_t::const_schedulable_ops_iterator_t itr = scheduler.begin_candidates();

  EXPECT_EQ(*(*itr), "u");
  ++itr;
  EXPECT_TRUE(itr == scheduler.end_candidates());
}
