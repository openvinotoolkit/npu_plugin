#include "gtest/gtest.h"
#include "feasible_scheduler.hpp"
#include <limits>
#include <unordered_map>

namespace scheduler_unit_tests {


// Simple DAG for unit testing the core algorithm //
class Operation_Dag {
  public:
    typedef Operation_Dag dag_t;
    typedef std::string operation_t;
    typedef std::vector<operation_t> adjacency_list_t;
    typedef typename adjacency_list_t::const_iterator const_adj_list_iterator_t;
    typedef std::unordered_map<operation_t, adjacency_list_t> adjacency_map_t;
    typedef typename adjacency_map_t::const_iterator const_adj_map_iterator_t;

    // resource cost model //
    typedef size_t resource_t;
    typedef std::unordered_map<operation_t, resource_t> resource_cost_model_t;
    // delay cost model //
    typedef size_t delay_t;
    typedef std::unordered_map<operation_t, delay_t> delay_cost_model_t;

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


    Operation_Dag(const adjacency_map_t& in) : adj_map_(in) {init();} 
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

    // Precondition: new delay model most have an entry for every op. //
    void reset_delay_model(const delay_cost_model_t& delay_model) {
      delay_cost_model_.clear();
      delay_cost_model_ = delay_model;
    }

    // Precondition: new delay model most have an entry for every op. //
    void reset_resource_model(const resource_cost_model_t& resource_model) {
      resource_cost_model_.clear();
      resource_cost_model_ = resource_model;
    }


    delay_t get_operation_delay(const operation_t& op) const {
      resource_cost_model_t::const_iterator itr = delay_cost_model_.find(op);
      assert(itr != delay_cost_model_.end());
      return itr->second;
    }

    resource_t get_operation_resources(const operation_t& op) const {
      resource_cost_model_t::const_iterator itr = resource_cost_model_.find(op);
      assert(itr != resource_cost_model_.end());
      return itr->second;
    }

    ////////////////////////////////////////////////////////////////////////////
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

    static delay_t delay(const dag_t& dag, const operation_t& op) {
      return dag.get_operation_delay(op);
    }

    static resource_t resource_utility(const dag_t& dag, const operation_t& op)
    {
      return dag.get_operation_resources(op);
    }

    ////////////////////////////////////////////////////////////////////////////

  private:

    void init() {
      reset_to_uniform_delay_model();
      reset_to_uniform_resource_model();
    }

    void reset_to_uniform_delay_model(delay_t d=1) {
      delay_cost_model_.clear();

      const_operation_iterator_t itr=begin(), itr_end=end();

      while (itr != itr_end) {
        delay_cost_model_[*itr] = delay_t(d);
        ++itr;
      }
    }

    void reset_to_uniform_resource_model(resource_t r=1) {
      resource_cost_model_.clear();

      const_operation_iterator_t itr=begin(), itr_end=end();

      while (itr != itr_end) {
        resource_cost_model_[*itr] = delay_t(r);
        ++itr;
      }
    }


    adjacency_map_t adj_map_;
    delay_cost_model_t delay_cost_model_;
    resource_cost_model_t resource_cost_model_;
}; // class Operation_Dag //

// Using the Cumulative_Resource_State //
typedef mv::lp_scheduler::Cumulative_Resource_State<
  Operation_Dag::resource_t, Operation_Dag::operation_t> resource_state_t;

} // namespace scheduler_unit_tests //

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits<scheduler_unit_tests::Operation_Dag>
  : public scheduler_unit_tests::Operation_Dag {
    ////////////////////////////////////////////////////////////////////////////
    // input graph model and delay model are used from Operation_Dag itself   //
    using scheduler_unit_tests::Operation_Dag::Operation_Dag;

    ////////////////////////////////////////////////////////////////////////////
    // define resource update model //
    typedef scheduler_unit_tests::resource_state_t resource_state_t;

    static void initialize_resource_upper_bound(const resource_t& upper_bound,
        resource_state_t& state) {
      state.initialize_resource_upper_bound(upper_bound);
    }

    static bool is_resource_available(const resource_t& demand,
          const resource_state_t& state) {
      return state.is_resource_available(demand);
    }

    static bool schedule_operation(const operation_t& op,
        const resource_t& demand, resource_state_t& state) {
      return state.assign_resources(op, demand);
    }

    static bool unschedule_operation(const operation_t& op,
        resource_state_t& state) {
      return state.unassign_resources(op);
    }
    ////////////////////////////////////////////////////////////////////////////

}; // specialization for scheduler_unit_tests::dag_t //

} // namespace lp_scheduler //
} // namespace mv //

using namespace scheduler_unit_tests;
typedef Operation_Dag dag_t;

// Unit tests to verify the iterator in the dag //
TEST(lp_scheduler_unit_test_infra, empty_graph) {
  dag_t::adjacency_map_t in = { }; 
  dag_t dag(in);

  EXPECT_TRUE(dag.begin() == dag.end());
  EXPECT_TRUE(dag.begin_edges() == dag.end());
}

TEST(lp_scheduler_unit_test_infra, edge_iteration_no_edges) {
  dag_t::adjacency_map_t in = { {"u", {}  }, {"v", {}  }, {"w", {} }};
  dag_t dag(in);

  dag_t::const_operation_iterator_t itr = dag.begin_edges(),
      itr_end = dag.end();

  EXPECT_TRUE(itr == itr_end);
}

TEST(lp_scheduler_unit_test_infra, edge_iteration_with_edges) {
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

TEST(lp_scheduler_unit_test_infra, vertex_iteration_no_edges) {
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

TEST(lp_scheduler_unit_test_infra, vertex_iteration_with_edges) {
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

TEST(lp_scheduler_unit_test_infra, outgoing_edges_of_node) {
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
TEST(lp_scheduler, chain_scheduler_unbounded_resources) {
  // create input //
  dag_t::adjacency_map_t in = { {"u", {"v"}  }, {"v", {"w"}  }, {"w", {"x"} },
      {"x", {} } }; // chain //
  dag_t g(in);

  // resource bound is 10 however chain dependency will force a serial schedule
  scheduler_t scheduler_begin(g, 10), schedule_end; 

  EXPECT_EQ(scheduler_begin.current_time(), 0UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);

  // operation "u" should be scheduled first //
  EXPECT_EQ(*scheduler_begin, "u");
  EXPECT_EQ(scheduler_begin.current_time(), 0UL);

  // next operation must be "v" at time 1 //
  ++scheduler_begin;
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "v");
  EXPECT_EQ(scheduler_begin.current_time(), 1UL);

  // next operation must be "w" at time 1 //
  ++scheduler_begin;
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "w");
  EXPECT_EQ(scheduler_begin.current_time(), 2UL);

  // next operation must be "w" at time 1 //
  ++scheduler_begin;
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "x");
  EXPECT_EQ(scheduler_begin.current_time(), 3UL);

  ++scheduler_begin;
  EXPECT_TRUE(scheduler_begin == schedule_end);
}


TEST(lp_scheduler, unfeasible_schedule) {
  // create input //
  dag_t::adjacency_map_t in = { {"u", {"v"}  }, {"v", {"w"}  }, {"w", {"x"} },
      {"x", {} } }; // chain //
  dag_t g(in);

  // resource bound is 0 no schedule is not possible //
  scheduler_t scheduler_begin(g, 0), schedule_end; 
  EXPECT_TRUE(scheduler_begin == schedule_end);
}


TEST(lp_scheduler, no_resource_bottle_neck) {
  /// input //
  dag_t::adjacency_map_t in = { 
    {"start", {"u", "v", "w", "x"} },
        {"u", {}  }, {"v", {}  }, {"w", {} }, {"x", {} } }; // fan //
  dag_t g(in);

  scheduler_t scheduler_begin(g, 100 /*resource bound*/), schedule_end; 

  EXPECT_EQ(scheduler_begin.current_time(), 0UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "start");

  ++scheduler_begin;
  // now its feasible to start all ops "u", "v", "w", "x" at
  // time step 1 since they don't exceed resource bound //
  EXPECT_EQ(scheduler_begin.current_time(), 1UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "u");

  ++scheduler_begin;
  // now the scheduler must still be at time step-1 //
  EXPECT_EQ(scheduler_begin.current_time(), 1UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "v");

  ++scheduler_begin;
  // now the scheduler must still be at time step-1 //
  EXPECT_EQ(scheduler_begin.current_time(), 1UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "w");

  ++scheduler_begin;
  // now the scheduler must still be at time step-1 //
  EXPECT_EQ(scheduler_begin.current_time(), 1UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "x");

  ++scheduler_begin;
  ASSERT_TRUE(scheduler_begin == schedule_end);
}

TEST(lp_scheduler, resource_bound_of_two_units) {
  /// input //
  dag_t::adjacency_map_t in = { 
    {"start", {"u", "v", "w", "x"} },
        {"u", {}  }, {"v", {}  }, {"w", {} }, {"x", {} } }; // fan //
  dag_t g(in);

  scheduler_t scheduler_begin(g, 2 /*resource bound*/), schedule_end; 

  EXPECT_EQ(scheduler_begin.current_time(), 0UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "start");

  ++scheduler_begin;
  // now its feasible to start atmost two ops at each time step//
  // time step 1 since they don't exceed resource bound //
  EXPECT_EQ(scheduler_begin.current_time(), 1UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "u");

  ++scheduler_begin;
  // now the scheduler must still be at time step-1 //
  EXPECT_EQ(scheduler_begin.current_time(), 1UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "v");

  ++scheduler_begin;
  // now the scheduler must still be at time step-2 //
  EXPECT_EQ(scheduler_begin.current_time(), 2UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "w");

  ++scheduler_begin;
  // now the scheduler must still be at time step-1 //
  EXPECT_EQ(scheduler_begin.current_time(), 2UL);
  ASSERT_FALSE(scheduler_begin == schedule_end);
  EXPECT_EQ(*scheduler_begin, "x");

  ++scheduler_begin;
  ASSERT_TRUE(scheduler_begin == schedule_end);
}

TEST(lp_scheduler, two_connected_diamonds) {
  dag_t::adjacency_map_t in = {
    {"a", {"b", "c"} }, {"b", {"d"}}, {"c", {"d"}},
    {"d", {"e", "f"}}, {"e", {"g"}}, {"f", {"g"}}, {"g", {}} };
  dag_t g(in);

  {
    scheduler_t scheduler_begin(g, 2), scheduler_end;
    
    while (scheduler_begin != scheduler_end) {
      ++scheduler_begin;
    }
    EXPECT_EQ(scheduler_begin.current_time(), 5UL);
  }

  // if we constrain to 1 resource the makespan increases to 7 //
  {
    scheduler_t scheduler_begin(g, 1), scheduler_end;
    
    while (scheduler_begin != scheduler_end) {
      ++scheduler_begin;
    }
    EXPECT_EQ(scheduler_begin.current_time(), 7UL);
  }
}

TEST(lp_scheduler, dependency_dominates_availability) {
  dag_t::adjacency_map_t in = { 
    {"start", {"u", "v", "w"} },
        {"u", {"x"}  }, {"v", {"x"}  }, {"w", {"x"} }, {"x", {} } }; 
  dag_t g(in);

  scheduler_t scheduler_begin(g, 2 /*resource bound*/), scheduler_end; 

  // notice that even though we have 2 resources available "x" cannot
  // be scheduled with any of "u", "v" or "w" it has to wait till all 
  // the three "u,v,w" complete.

  size_t itr_count = 0UL;
  size_t op_time_of_x = 0UL;
  while (scheduler_begin != scheduler_end) {
    if (*scheduler_begin == "x") {
      op_time_of_x = scheduler_begin.current_time();
    }
    ++itr_count;
    ++scheduler_begin;
    ASSERT_TRUE(itr_count <= 5);
  }
  EXPECT_EQ(op_time_of_x, 3UL);
  EXPECT_EQ(scheduler_begin.current_time(), 4UL);
}




