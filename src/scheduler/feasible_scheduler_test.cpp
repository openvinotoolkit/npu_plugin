#include <iterator>
#include <limits>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "scheduler/feasible_scheduler.hpp"

namespace scheduler_unit_tests {


// Simple DAG for unit testing the core algorithm //
class Operation_Dag {
  public:
    typedef Operation_Dag dag_t;
    typedef std::string operation_t;
    typedef std::hash<std::string> operation_hash_t;
    typedef std::vector<operation_t> adjacency_list_t;
    typedef typename adjacency_list_t::const_iterator const_adj_list_iterator_t;
    typedef std::unordered_map<operation_t, adjacency_list_t> adjacency_map_t;
    typedef typename adjacency_map_t::const_iterator const_adj_map_iterator_t;

    // resource cost model //
    typedef size_t resource_t;
    typedef std::unordered_map<operation_t, resource_t> resource_cost_model_t;
    typedef std::unordered_set<operation_t> data_op_set_t;
    // delay cost model //
    typedef size_t delay_t;
    typedef std::unordered_map<operation_t, delay_t> delay_cost_model_t;

    class const_operation_iterator_t {
      public:
        const_operation_iterator_t() : adj_map_ptr_(NULL),
          itr_begin_(), itr_end_(), itr_adj_begin_(), itr_adj_end_(),
          iterate_edges_(false) {}

        const_operation_iterator_t(const const_operation_iterator_t& o)
          : adj_map_ptr_(), itr_begin_(), itr_end_(), itr_adj_begin_(),
            itr_adj_end_(), iterate_edges_(false) {
            adj_map_ptr_ = o.adj_map_ptr_;
            itr_begin_ = o.itr_begin_;
            itr_end_ = o.itr_end_;
            itr_adj_begin_ = o.itr_adj_begin_;
            itr_adj_end_ = o.itr_adj_end_;
            iterate_edges_ = o.iterate_edges_;
         }

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

    const_operation_iterator_t begin_in(const operation_t& op) const {
      adjacency_map_t::const_iterator itr = inv_adj_map_.find(op), itr_next;
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

    const operation_t& get_operation(const std::string& name) const {
      adjacency_map_t::const_iterator itr = adj_map_.find(name);
      assert(itr != adj_map_.end());
      return itr->first;
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

    size_t size() const { return adj_map_.size(); }

    bool is_data_op(const operation_t& op) const {
      return !(data_op_set_.find(op) == data_op_set_.end());
    }

    void reset_data_op_set(const data_op_set_t& in) { data_op_set_ = in; }

    ////////////////////////////////////////////////////////////////////////////
    // scheduler_traits //

    static const char * operation_name(const operation_t& op) {
      return op.c_str();
    }
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

    static const_operation_iterator_t incoming_operations_begin(const dag_t& g,
        const operation_t& op) { return g.begin_in(op); }

    static const_operation_iterator_t incoming_operations_end(const dag_t& g,
        const operation_t& op) { return g.end(op); }

    static delay_t delay(const dag_t& dag, const operation_t& op) {
      return dag.get_operation_delay(op);
    }

    // TODO(vamsikku) : change this to specific values of spilled delay //
    static delay_t spilled_read_delay(const dag_t& , const operation_t& ) {
      return delay_t(1);
    }

    static delay_t spilled_write_delay(const dag_t& , const operation_t& ){
      return delay_t(1);
    }

    static resource_t resource_utility(const dag_t& dag, const operation_t& op)
    {
      return dag.get_operation_resources(op);
    }

    static bool is_data_operation(const dag_t& dag, const operation_t& op) {
      return dag.is_data_op(op);
    }

    static bool is_compute_operation(const dag_t& dag, const operation_t& op) {
      return !dag.is_data_op(op);
    }

    ////////////////////////////////////////////////////////////////////////////

  private:

    void init() {
      reset_to_uniform_delay_model();
      reset_to_uniform_resource_model();
      init_inv_adj_map();
    }

    // Precondition: adj_map_ must be constructed //
    void init_inv_adj_map() {
      inv_adj_map_.clear();
      const_adj_map_iterator_t itr = adj_map_.begin(), itr_end = adj_map_.end();

      for (; itr != itr_end; ++itr) {
        const_adj_list_iterator_t inv_itr = (itr->second).begin(),
                                  inv_itr_end = (itr->second).end();
        for (; inv_itr != inv_itr_end; ++inv_itr) {
          inv_adj_map_[ *inv_itr ].push_back( itr->first);
        }
      }
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
    adjacency_map_t inv_adj_map_; // all incoming edges of a node //
    delay_cost_model_t delay_cost_model_;
    resource_cost_model_t resource_cost_model_;
    data_op_set_t data_op_set_;
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

    static bool is_empty_demand(const resource_t& demand) {
      return (demand == resource_t(0UL));
    }

    static bool is_resource_available(const resource_t& demand,
          const resource_state_t& state) {
      return state.is_resource_available(demand);
    }

    static bool schedule_operation(const operation_t& op,
        const resource_t& demand, resource_state_t& state,
        const_operation_iterator_t, const_operation_iterator_t) {
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

bool GetIncomingEdges(const dag_t::operation_t& op, const dag_t& dag,
    dag_t::adjacency_map_t& nodes) {
  nodes.clear();

  dag_t::const_operation_iterator_t itr = dag.begin_in(op), itr_end;
  while (itr != itr_end) {
    if (nodes.find(*itr) != nodes.end()) {
      // duplicate in coming node //
      return false;
    }
    nodes.insert(std::make_pair(*itr, dag_t::adjacency_list_t()));
    ++itr;
  }
  return true;
}

TEST(lp_scheduler_unit_test_infra, vertex_iteration_with_incoming_edges) {
  dag_t::adjacency_map_t in = { {"u", {"e", "f", "g"}  },
      {"v", {"e"}  }, {"w", {"e", "f"} }, {"e", {}}, {"f", {}}, {"g", {}} };
  dag_t dag(in);

  {
    // incoming nodes of e = {v, w, u } //
    dag_t::adjacency_map_t expected = { {"v", {}}, {"w", {}}, {"u", {}} };
    dag_t::adjacency_map_t found;

    ASSERT_TRUE(GetIncomingEdges("e", dag, found));
    EXPECT_EQ(found, expected);
  }

  {
    // incoming nodes of f = { w, u } //
    dag_t::adjacency_map_t expected = { {"w", {}}, {"u", {}} };
    dag_t::adjacency_map_t found;

    ASSERT_TRUE(GetIncomingEdges("f", dag, found));
    EXPECT_EQ(found, expected);
  }


  {  // no incoming edges for u and v //
    ASSERT_TRUE(dag.begin_in("u") == dag.end());
    ASSERT_TRUE(dag.begin_in("v") == dag.end());
  }
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
typedef scheduler_t cumulative_memory_scheduler_t;

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

// Unit tests for testing the contiguous resource state //
typedef mv::lp_scheduler::Contiguous_Resource_State<size_t, std::string>
  contiguous_resource_state_t;

TEST(Contiguous_Resource_State, unit_resource_upper_bound) {
  std::vector<std::string> ops = {"op1", "op2"};

  contiguous_resource_state_t rstate(1);

  EXPECT_TRUE(rstate.is_resource_available(1));
  EXPECT_FALSE(rstate.is_resource_available(2));

  // now make a transaction //
  EXPECT_TRUE(rstate.assign_resources(ops[0], 1));

  // since the resource is already used this should be false //
  EXPECT_FALSE(rstate.is_resource_available(1));
}

TEST(Contiguous_Resource_State, cumulative_available_but_not_contiguous) {
  std::vector<std::string> ops = {"#", "*", "%"};

  contiguous_resource_state_t rstate(10);
  /*start state = [----------] */

  EXPECT_TRUE(rstate.is_resource_available(5));
  // now make a transaction //
  EXPECT_TRUE(rstate.assign_resources(ops[0], 5));
  /* state after = [#####-----]*/

  EXPECT_TRUE(rstate.is_resource_available(2));
  // now make a transaction //
  EXPECT_TRUE(rstate.assign_resources(ops[1], 2));
  /* state after = [#####**---]*/

  EXPECT_FALSE(rstate.is_resource_available(5));
  EXPECT_TRUE(rstate.is_resource_available(1));
  EXPECT_TRUE(rstate.is_resource_available(2));
  EXPECT_TRUE(rstate.is_resource_available(2));

  // now unassign ops[0]=# //
  EXPECT_TRUE(rstate.unassign_resources(ops[0]));
  EXPECT_TRUE(rstate.is_resource_available(5));
  // sum of free fragments is 8units but contiguous is max of 5 units //
  EXPECT_FALSE(rstate.is_resource_available(8));

  // now we unassign op[1]=* then we should have contiguous 8 units//
  EXPECT_TRUE(rstate.unassign_resources(ops[1]));
  EXPECT_TRUE(rstate.is_resource_available(8));
  EXPECT_TRUE(rstate.assign_resources(ops[2], 8));
  EXPECT_FALSE(rstate.is_resource_available(5));
}

TEST(Contiguous_Resource_State, verify_interval_weak_ordering) {
  contiguous_resource_state_t::interval_info_t a(0,9), b(1, 1), c(5,5);
  contiguous_resource_state_t::interval_length_ordering_t weak_order;

  EXPECT_FALSE(weak_order(a,a));
  EXPECT_TRUE(weak_order(a,b)); EXPECT_FALSE(weak_order(b,a));
  EXPECT_TRUE(weak_order(a,c)); EXPECT_FALSE(weak_order(c,a));
  EXPECT_FALSE(weak_order(b,c)); EXPECT_FALSE(weak_order(c,b));
}


TEST(Contiguous_Resource_State, simultaneous_resource_availablity) {

  std::vector<std::string> ops = {"#", "*", "%"};

  contiguous_resource_state_t rstate(10);

  ASSERT_TRUE(rstate.is_resource_available(9));
  ASSERT_TRUE(rstate.is_resource_available(10));
  ASSERT_FALSE(rstate.is_resource_available(11));

  {
    size_t demands[3] = {7,1,1};
    EXPECT_TRUE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }
  {
    size_t demands[3] = {7,1,3};
    EXPECT_FALSE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }
}

TEST(Contiguous_Resource_State, simultaneous_resource_availablity_pack_info) {
  typedef typename contiguous_resource_state_t::interval_info_t interval_info_t;

  std::vector<std::string> ops = {"#", "*", "%"};

  contiguous_resource_state_t rstate(10);

  ASSERT_TRUE(rstate.is_resource_available(9));
  ASSERT_TRUE(rstate.is_resource_available(10));
  ASSERT_FALSE(rstate.is_resource_available(11));

  {
    std::vector<interval_info_t> found_packing;
    size_t demands[4] = {7,1,1};

    EXPECT_TRUE(rstate.pack_demands_into_free_bins(demands, demands+4,
            std::back_inserter(found_packing) ));

    std::vector<interval_info_t> expected_packing = { {1, 7}, {8,8}, {9,9} };
    EXPECT_EQ(expected_packing, found_packing);
  }

  {
    size_t demands[3] = {7,1,3};
    EXPECT_FALSE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }

}

TEST(Contiguous_Resource_State, simultaneous_resource_availablity_typical) {

  std::vector<std::string> ops = {"#", "*", "%"};

  contiguous_resource_state_t rstate(10);


  EXPECT_TRUE(rstate.assign_resources(ops[0], 1));
  {
    size_t demands[3] = {7,1,1};
    EXPECT_TRUE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }
  {
    size_t demands[3] = {7,1,2};
    EXPECT_FALSE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }

  EXPECT_TRUE(rstate.assign_resources(ops[1], 2));
  EXPECT_TRUE(rstate.unassign_resources(ops[0]));

  {
    // total of 8 units available {1 , 7} //

    size_t demands[3] = {7,1,0};
    EXPECT_TRUE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }

  {
    // total of 8 units available {1 , 7} but you cannot pack {6,2} into 
    // bins of sizes {1, 7} //

    size_t demands[3] = {6,0,2};
    EXPECT_FALSE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }

  {
    // total of 8 units available {1 , 7} //

    size_t demands[3] = {8,0,0};
    EXPECT_FALSE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }

}

TEST(Contiguous_Resource_State, unit_simultaneous_resource_availablity) {

  std::vector<std::string> ops = {"#", "*", "%"};

  contiguous_resource_state_t rstate(1);

  {
    size_t demands[3] = {0,1,0};
    EXPECT_TRUE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }
  {
    size_t demands[3] = {1,1,0};
    EXPECT_FALSE(
        rstate.are_resources_available_simultaneously(demands, demands+3));
  }
  {
    size_t demands[5] = {0,0,0,0,0};
    EXPECT_TRUE(
        rstate.are_resources_available_simultaneously(demands, demands+5));
  }
}

TEST(Contiguous_Resource_State, active_resources_lookup) {

  std::vector<std::string> ops = {"#", "*", "%"};

  contiguous_resource_state_t rstate(10);

  ASSERT_TRUE(rstate.is_resource_available(3));
  ASSERT_TRUE(rstate.assign_resources(ops[1], 3));
  ASSERT_FALSE(rstate.is_key_using_resources(ops[0]));
  ASSERT_FALSE(rstate.is_key_using_resources(ops[2]));

  ASSERT_TRUE(rstate.unassign_resources(ops[1]));
  ASSERT_FALSE(rstate.is_key_using_resources(ops[1]));
}


////////////////////////////////////////////////////////////////////////////////
// Test scheduler with contiguous resource model //
namespace scheduler_unit_tests {

typedef mv::lp_scheduler::Contiguous_Resource_State<Operation_Dag::resource_t,
        Operation_Dag::operation_t> memory_resource_state_t;

struct operation_dag_with_contiguous_resource_model_t;

} // namespace scheduler_unit_tests //

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits<
  scheduler_unit_tests::operation_dag_with_contiguous_resource_model_t
  > : public scheduler_unit_tests::Operation_Dag {
    ////////////////////////////////////////////////////////////////////////////
    // input graph model and delay model are used from Operation_Dag itself   //
    using scheduler_unit_tests::Operation_Dag::Operation_Dag;

    ////////////////////////////////////////////////////////////////////////////
    // define resource update model //
    typedef scheduler_unit_tests::memory_resource_state_t resource_state_t;

    static void initialize_resource_upper_bound(const resource_t& upper_bound,
        resource_state_t& state) {
      state.initialize_resource_upper_bound(upper_bound);
    }

    static bool is_resource_available(const resource_t& demand,
          const resource_state_t& state) {
      return state.is_resource_available(demand);
    }

    static bool schedule_operation(const operation_t& op,
        const resource_t& demand, resource_state_t& state,
        const_operation_iterator_t, const_operation_iterator_t) {
      return state.assign_resources(op, demand);
    }

    static bool unschedule_operation(const operation_t& op,
        resource_state_t& state) {
      return state.unassign_resources(op);
    }
    ////////////////////////////////////////////////////////////////////////////

}; // specialization for scheduler_unit_tests::dag_t //

} // namespace lp_scheduler
} // namespace mv

typedef mv::lp_scheduler::Feasible_Schedule_Generator<
  operation_dag_with_contiguous_resource_model_t > memory_scheduler_t; 

TEST(memory_lp_scheduler, basic_test) {
  /*
  ///////////////////////////////////////////
  //       +-->-(a)\
  //       |        \
  //       |         \
  //       |          (c)
  // (start)-->-(b)  /
  //       |        /
  //       |       /
  //       +--->(d)
  //
  // d(a) = d(d) = d(c) 1 , d(b) = 3
  // r(a) = 3, r(b) = 3 , r(d) = 4, r(c) = 5
  ///////////////////////////////////////////
  */

  dag_t::adjacency_map_t in = {
    {"start", {"a", "b", "d"} }, {"a", {"c"}}, {"b", {}}, {"d", {"c"}},
      {"c", {}} };
  dag_t::delay_cost_model_t delay = { {"start", 1},
    {"a", 1}, {"d", 1}, {"c", 1}, {"b", 3} };
  dag_t::resource_cost_model_t memory = { {"start", 1},
    {"a", 3}, {"b", 3}, {"d", 4}, {"c", 5} };

  dag_t g(in);

  g.reset_delay_model(delay);
  g.reset_resource_model(memory);

  // scheduler with 10 units for resources//
  memory_scheduler_t scheduler_begin(g, 10), scheduler_end; 

  size_t itr_count = 0UL;
  size_t start_time_of_c = 0UL;
  while (scheduler_begin != scheduler_end) {
    std::cout << "op=" << *scheduler_begin << " time=" 
      << scheduler_begin.current_time() << std::endl;
    if (*scheduler_begin  == "c") {
      start_time_of_c = scheduler_begin.current_time();
    }
    ++itr_count;
    ++scheduler_begin;
    ASSERT_TRUE(itr_count <= 5);
  }
  EXPECT_EQ(start_time_of_c, 4UL);
  // completion time //
  EXPECT_EQ(scheduler_begin.current_time(), 5UL);
}


TEST(memory_lp_scheduler, cumulative_version) {
  // Same as the above test however we relax the cumulative resource constraint
  // notice the makespan improvement form 5units to 4units. Also the operation
  // 'c' now is being scheduled at time of 2units. The operation 'b' is now on
  // the critical path.

  dag_t::adjacency_map_t in = {
    {"start", {"a", "b", "d"} }, {"a", {"c"}}, {"b", {}}, {"d", {"c"}},
      {"c", {}} };
  dag_t::delay_cost_model_t delay = { {"start", 1},
    {"a", 1}, {"d", 1}, {"c", 1}, {"b", 3} };
  dag_t::resource_cost_model_t memory = { {"start", 1},
    {"a", 3}, {"b", 3}, {"d", 4}, {"c", 5} };

  dag_t g(in);

  g.reset_delay_model(delay);
  g.reset_resource_model(memory);

  // scheduler with 10 units for resources//
  cumulative_memory_scheduler_t scheduler_begin(g, 10), scheduler_end; 

  size_t itr_count = 0UL;
  size_t start_time_of_c = 0UL;
  while (scheduler_begin != scheduler_end) {
    std::cout << "op=" << *scheduler_begin << " time=" 
      << scheduler_begin.current_time() << std::endl;
    if (*scheduler_begin  == "c") {
      start_time_of_c = scheduler_begin.current_time();
    }
    ++itr_count;
    ++scheduler_begin;
    ASSERT_TRUE(itr_count <= 5);
  }
  EXPECT_EQ(start_time_of_c, 2UL);
  EXPECT_EQ(scheduler_begin.current_time(), 4UL);
}

typedef mv::lp_scheduler::Producer_Consumer_Contiguous_Resource<size_t,
          std::string>  contiguous_producer_consumer_resource_t;

// Unit tests for Producer_Consumer_Contiguous_Resource state //
TEST(Producer_Consumer_Contiguous_Resource_State, basic_test) {
  contiguous_producer_consumer_resource_t resource_state(10UL);

  EXPECT_TRUE(resource_state.is_resource_available(1UL));
  EXPECT_FALSE(resource_state.is_resource_available(11UL));
}


TEST(Producer_Consumer_Contiguous_Resource_State, simple_dependency) {
  contiguous_producer_consumer_resource_t resource_state(10UL);

  //  a --> {b, c} //
  //  f --> {g}
  //  
  //  r(a) = 6 , r(f) = 5 , r(b)=2, r(c)=2, r(d)=4, r(g)=2//
  std::vector<std::string> consumers_of_a = {"b", "c"}, consumers;
  std::unordered_map<std::string, size_t> r = { {"a", 6UL}, {"f", 5UL},
    {"b", 2UL}, {"c", 2UL}, {"g", 2UL} };

  EXPECT_TRUE(resource_state.is_resource_available(r["a"]));
  EXPECT_TRUE(resource_state.is_resource_available(r["f"]));

  // resources of "a" are engaged until "b" , "c" and "d" unassign resources //
  EXPECT_TRUE(resource_state.assign_resources("a", r["a"],
        consumers_of_a.begin(), consumers_of_a.end()));

  EXPECT_TRUE(resource_state.is_resource_available(r["b"]));
  EXPECT_TRUE(resource_state.assign_resources("b", r["b"],
        consumers.end(), consumers.end()));

  EXPECT_TRUE(resource_state.is_resource_available(r["c"]));
  EXPECT_TRUE(resource_state.assign_resources("c", r["c"],
        consumers.end(), consumers.end()));

  // current 5+2+2=9units of resource is engaged //


  // NOTE: we dont have 5units of contigous resource //
  EXPECT_FALSE(resource_state.is_resource_available(r["f"]));

  // unassign resources to "a" but its resources will still be engaged until "b"
  // and "c" have unassigned resources. Hence resource request for "d" will
  // still fail.
  EXPECT_TRUE(resource_state.unassign_resources("a"));
 
  // Since the consumers of "a" have not yet been finished. //
  EXPECT_FALSE(resource_state.is_resource_available(r["f"]));

  // unassign resources of "b" //
  EXPECT_TRUE(resource_state.unassign_resources("b"));

  // since both "a" and "c" are using resources //
  EXPECT_FALSE(resource_state.is_resource_available(r["f"]));

  EXPECT_TRUE(resource_state.unassign_resources("c"));

  // now all resources are available //
  EXPECT_TRUE(resource_state.is_resource_available(10UL));
}

TEST(Producer_Consumer_Contiguous_Resource_State,
    dependency_chain_cascade_deallocation) {

  std::vector<std::string> ops = {"a", "b", "c", "d"};
  std::vector<std::string>::const_iterator ops_itr = ops.begin();
  std::vector<size_t> r = {5UL, 3UL, 1UL, 1UL};
  contiguous_producer_consumer_resource_t resource_state(10UL);


  // chain dependency: a->b->c->d //
  ASSERT_TRUE(resource_state.is_resource_available(10UL));
  ASSERT_FALSE(resource_state.is_resource_available(11UL));

  // "a" keeps 5units engaged till "b" is finished //
  ASSERT_TRUE(resource_state.assign_resources(ops[0], r[0],
        ops_itr+1, ops_itr+2));
  ASSERT_TRUE(resource_state.unassign_resources(ops[0]));
  // ref_count[a] = 1 //

  ASSERT_FALSE(resource_state.is_resource_available(6UL));
  ASSERT_TRUE(resource_state.is_resource_available(5UL));

  ASSERT_TRUE(resource_state.assign_resources(ops[1], r[1],
        ops_itr+2, ops_itr+3));
  // ref_count[a] = 1, ref_count[b] = 2 //
  ASSERT_TRUE(resource_state.unassign_resources(ops[1]));
  // ref_count[a] = 0, ref_count[b] = 1 we get 5 resources from "a" //

  ASSERT_TRUE(resource_state.is_resource_available(5UL));

  ASSERT_TRUE(resource_state.assign_resources(ops[2], r[2],
        ops_itr+3, ops_itr+4));
  // ref_count[b] = 1, ref_count[c] = 2//
  ASSERT_TRUE(resource_state.unassign_resources(ops[2]));
  // ref_count[b] = 0, ref_count[c] = 1 we get 3 resources from "b" //

  ASSERT_TRUE(resource_state.is_resource_available(3UL));

  ASSERT_TRUE(resource_state.assign_resources(ops[3], r[3],
        ops.end(), ops.end()));
  // ref_count[c] = 1, ref_count[d] = 1 //
  ASSERT_TRUE(resource_state.unassign_resources(ops[3]));
  // ref_count[c] = 0, ref_count[d] = 0 //

  ASSERT_TRUE(resource_state.is_resource_available(10UL));
}

////////////////////////////////////////////////////////////////////////////////
// Test scheduler with contiguous resource model with producer and consumers//
namespace scheduler_unit_tests {

typedef mv::lp_scheduler::Producer_Consumer_Contiguous_Resource<
    Operation_Dag::resource_t, Operation_Dag::operation_t>
      producer_consumer_memory_resource_state_t;

struct operation_dag_with_producer_consumer_model_t;

} // namespace scheduler_unit_tests //

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits<
  scheduler_unit_tests::operation_dag_with_producer_consumer_model_t
  > : public scheduler_unit_tests::Operation_Dag {
    ////////////////////////////////////////////////////////////////////////////
    // input graph model and delay model are used from Operation_Dag itself   //
    using scheduler_unit_tests::Operation_Dag::Operation_Dag;

    ////////////////////////////////////////////////////////////////////////////
    // define resource update model //
    typedef scheduler_unit_tests::producer_consumer_memory_resource_state_t
        resource_state_t;

    static void initialize_resource_upper_bound(const resource_t& upper_bound,
        resource_state_t& state) {
      state.initialize_resource_upper_bound(upper_bound);
    }

    static bool is_resource_available(const resource_t& demand,
          const resource_state_t& state) {
      return state.is_resource_available(demand);
    }

    static bool schedule_operation(const operation_t& op,
        const resource_t& demand, resource_state_t& state,
        const_operation_iterator_t op_begin,
        const_operation_iterator_t op_end) {

      return (op == "start") ?
        state.assign_resources(op, demand, op_end, op_end) :
        state.assign_resources(op, demand, op_begin, op_end);
    }

    static bool unschedule_operation(const operation_t& op,
        resource_state_t& state) {
      return state.unassign_resources(op);
    }
    ////////////////////////////////////////////////////////////////////////////

}; // specialization for scheduler_unit_tests::dag_t //

} // namespace lp_scheduler
} // namespace mv

typedef mv::lp_scheduler::Feasible_Schedule_Generator<
  operation_dag_with_producer_consumer_model_t > producer_consumer_scheduler_t; 

TEST(lp_scheduler_producer_consumer, basic_test) {
  producer_consumer_scheduler_t schedule_begin, schedule_end;

  EXPECT_TRUE(schedule_begin == schedule_end);
}

TEST(lp_scheduler_producer_consumer, dependency_test) {
  dag_t::adjacency_map_t in =  { {"start", {"a", "d"} },
    {"a", {"b", "c" }}, {"b", {}}, {"c", {}}, {"d", {}} };

  // since "a" generates output of "b" and "c" its memory resources
  // are locked until both "b" and "c" are finished //

  dag_t::delay_cost_model_t delay = { {"start", 1},
    {"a", 1}, {"d", 1}, {"c", 10}, {"b", 12} };
  
  dag_t::resource_cost_model_t memory = { {"start", 1},
    {"a", 6}, {"b", 1}, {"d", 5}, {"c", 1} };

  dag_t g(in);

  g.reset_delay_model(delay);
  g.reset_resource_model(memory);

  producer_consumer_scheduler_t schedule_begin(g, 10), schedule_end;

  size_t iter_count = 0UL; 
  std::unordered_map<std::string, size_t> start_times;

  while (schedule_begin != schedule_end) {
    ++iter_count;
    ASSERT_TRUE(iter_count <= 5UL);
    ASSERT_TRUE(start_times.find(*schedule_begin) == start_times.end());
    start_times.insert(std::make_pair(*schedule_begin,
                                       schedule_begin.current_time()));
    ++schedule_begin;
  }

  // scheduler should produce output for each operation //
  for (auto itr = in.begin(); itr != in.end(); ++itr) {
    ASSERT_TRUE(start_times.find( itr->first ) != start_times.end());
  }

  // verify precedence constraints //
  ASSERT_TRUE(start_times["start"] < start_times["a"]);
  ASSERT_TRUE(start_times["a"] < start_times["b"]);
  ASSERT_TRUE(start_times["a"] < start_times["c"]);

  // if "a" starts before "d" then start time of "d" must be atleast 14 
  // due producer and consumer issue (resources of "a" are locked until
  // "b" and "c" are finished)
  ASSERT_TRUE(  (start_times["a"] >= start_times["d"]) ||

                (start_times["d"] >= 14UL) );
}

////////////////////////////////////////////////////////////////////////////////
typedef mv::lp_scheduler::Feasible_Memory_Schedule_Generator<Operation_Dag>
  feasible_memory_scheduler_t;

class Test_Fixture_Feasible_Memory_Scheduler
  : public feasible_memory_scheduler_t, public testing::Test {
  protected:
    typedef feasible_memory_scheduler_t::heap_t heap_t;
    typedef feasible_memory_scheduler_t::heap_element_t heap_element_t;
    typedef feasible_memory_scheduler_t::operation_t operation_t;
    typedef feasible_memory_scheduler_t::ready_data_list_t ready_data_list_t;
    typedef feasible_memory_scheduler_t::op_list_t op_list_t;
    typedef feasible_memory_scheduler_t::resource_t resource_t;
    typedef feasible_memory_scheduler_t::scheduled_op_info_list_t
        scheduled_op_info_list_t;
    typedef feasible_memory_scheduler_t::op_type_e op_type_E;
    typedef feasible_memory_scheduler_t::completion_time_ordering_t
        completion_time_ordering_t;

    void SetUp() override {}
    void TearDown() override {}

}; // class Test_Fixture_Feasible_Memory_Scheduler //

TEST_F(Test_Fixture_Feasible_Memory_Scheduler, test_heap_ordering) {
  operation_t o1("a"), o2("b"), o3("c");
  heap_element_t e1(o1, 4), e2(o2, 1 ), e3(o3, 7);
  completion_time_ordering_t order;

  EXPECT_TRUE(order(e1, e2)); // use the time //

  EXPECT_FALSE(order(e2, e1)); 
  EXPECT_FALSE(order(e1, e1));
}

TEST_F(Test_Fixture_Feasible_Memory_Scheduler, heap_push_and_pop) {
  operation_t o1("a"), o2("b"), o3("c");
  heap_element_t e1(o1, 4), e2(o2, 1 ), e3(o3, 7);
  const heap_element_t *top_elem = NULL;


  {
    EXPECT_TRUE(top_element() == NULL);

    push_to_heap(e1);
    top_elem = top_element();
    ASSERT_EQ(top_elem->time_, 4UL);
    ASSERT_EQ(top_elem->op_, e1.op_);
  }

  {

    ASSERT_TRUE(top_element() != NULL);
    push_to_heap(e2);
    ASSERT_TRUE(top_element()!= NULL);

    top_elem = top_element();
    ASSERT_EQ(top_elem->time_, 1UL);
    ASSERT_EQ(top_elem->op_, e2.op_);

    push_to_heap(e3);
    top_elem = top_element();
    ASSERT_EQ(top_elem->time_, 1UL);
    ASSERT_EQ(top_elem->op_, e2.op_);
  }

  {
    heap_element_t e = pop_from_heap();
    ASSERT_TRUE((e.op_ == e2.op_) && (e.time_ == e2.time_));

    ASSERT_TRUE(top_element()!= NULL);
    top_elem = top_element();
    ASSERT_EQ(top_elem->time_, 4UL);
    ASSERT_EQ(top_elem->op_, e1.op_);

    e = pop_from_heap();
    ASSERT_TRUE((e.op_ == e1.op_) && (e.time_ == e1.time_));

    ASSERT_TRUE(top_element()!= NULL);
    top_elem = top_element();
    ASSERT_EQ(top_elem->time_, 7UL);
    ASSERT_EQ(top_elem->op_, e3.op_);

    e = pop_from_heap();
    ASSERT_TRUE((e.op_ == e3.op_) && (e.time_ == e3.time_));
    ASSERT_TRUE(heap_empty());
  }
}

TEST_F(Test_Fixture_Feasible_Memory_Scheduler, pop_all_elements_at_this_time) {
  heap_element_t e[5UL] = { {"o1", 7}, {"o2", 7}, {"o3", 1},
      {"o4", 8}, {"o5", 9}};

  size_t n = 5UL;
  for (size_t i=0; i<n; i++) { push_to_heap(e[i]); }

  std::vector<heap_element_t> popped;
  pop_all_elements_at_this_time( 7UL, std::back_inserter(popped) );
  ASSERT_TRUE(popped.empty()); // since the current time is 1 //

  pop_from_heap(); 
  pop_all_elements_at_this_time( 7UL, std::back_inserter(popped) );
  std::vector<heap_element_t> expected = { {"o1", 7}, {"o2", 7} };
  EXPECT_EQ(expected, popped);

  ASSERT_TRUE(top_element() != NULL);
  EXPECT_EQ(top_element()->time_, 8UL);
  EXPECT_EQ(top_element()->op_, "o4");
}


TEST_F(Test_Fixture_Feasible_Memory_Scheduler, test_compute_ready_lists) {
  dag_t::adjacency_map_t in = { {"Input", {"A", "B", "C"} } ,
    {"A", {"B", "C"}}, {"B", {"C"}}, {"C", {}}, 
    {"A_input", {"A"}}, {"B_input", {"B"}}, {"C_input", {"C"}} };
  dag_t::data_op_set_t data_ops ={"A_input", "B_input", "C_input"};
  dag_t dag(in);

  dag.reset_data_op_set(data_ops);

  feasible_memory_scheduler_t::reset_input(dag);
  feasible_memory_scheduler_t::compute_op_in_degree();
  feasible_memory_scheduler_t::compute_ready_data_list();

  ready_data_list_t ready_data = { "A_input", "C_input", "B_input" };
  
  EXPECT_EQ(ready_data, ready_data_list_);

  // verify the in-degree of the other nodes in updated network // 
  EXPECT_EQ(get_op_in_degree("A"), 1UL);
  EXPECT_EQ(get_op_in_degree("B"), 2UL);
  EXPECT_EQ(get_op_in_degree("C"), 3UL);
  EXPECT_EQ(get_op_in_degree("Input"), 0UL);

  // there is only one ready operation at this time which is the Input //
  feasible_memory_scheduler_t::compute_ready_compute_list();
  EXPECT_TRUE(ready_active_list_.empty());
  EXPECT_FALSE(ready_list_.empty());
  EXPECT_EQ(ready_list_.size(), 1UL);

  {
    op_list_t expected = { "Input" };
    EXPECT_EQ(expected, ready_list_);
  }
}


TEST_F(Test_Fixture_Feasible_Memory_Scheduler, test_compute_in_degree) {
  dag_t::adjacency_map_t in = { {"Input", {"A", "B", "C"} } ,
    {"A", {"B", "C"}}, {"B", {"C"}}, {"C", {}}, 
    {"A_input", {"A"}}, {"B_input", {"B"}}, {"C_input", {"C"}} };
  dag_t::data_op_set_t data_ops ={"A_input", "B_input", "C_input"};
  dag_t dag(in);

  dag.reset_data_op_set(data_ops);

  feasible_memory_scheduler_t::reset_input(dag);
  feasible_memory_scheduler_t::compute_op_in_degree();

  EXPECT_EQ(get_op_in_degree("Input"), 0UL);
  EXPECT_EQ(get_op_in_degree("A"), 2UL);
  EXPECT_EQ(get_op_in_degree("B"), 3UL);
  EXPECT_EQ(get_op_in_degree("C"), 4UL);
}

TEST_F(Test_Fixture_Feasible_Memory_Scheduler, is_ready_op_schedulable) {
  dag_t::adjacency_map_t in = { {"Input", {"A", "B"} }, {"A", {"B"}},
    {"B", {}}, {"A_in", {"A"} }, {"B_in", {"B"} } };
  dag_t::resource_cost_model_t memory = { {"Input", 3UL}, {"A", 3UL},
    {"A_in",4UL}, {"B_in", 2UL}, {"B", 2UL} };

  dag_t g(in);
  g.reset_resource_model(memory);

  reset(g, resource_t(10UL));

  EXPECT_TRUE(is_ready_compute_operation_schedulable("Input"));
  EXPECT_TRUE(schedule_compute_op("Input"));

  // now get the demand list of operation "A" and operation "B" since
  // both depend on the input. The demand of A = (4 + 3 + 3) - 3 and
  // the demand of B = (3 + 2 + 2 + 3) - 3 the available demand is 7 
  {
    std::vector<resource_t> demand_list;
    std::vector<resource_t> expected_list = {3, 4};

    ASSERT_EQ(get_non_empty_op_demand_list("A",
            std::back_inserter(demand_list)), 2UL);
    std::sort(demand_list.begin(), demand_list.end());
    EXPECT_EQ(demand_list, expected_list);
  }


  {
    std::vector<resource_t> demand_list;
    std::vector<resource_t> expected_list = {2, 2, 3};

    ASSERT_EQ(get_non_empty_op_demand_list("B",
            std::back_inserter(demand_list)), 3UL);
    std::sort(demand_list.begin(), demand_list.end());
    EXPECT_EQ(demand_list, expected_list);
  }

  // check the heap //
  {
    const heap_element_t *top_elem = NULL;
    heap_element_t expected("Input", 0);

    ASSERT_TRUE(top_element_start_time());
    top_elem = top_element_start_time();
    EXPECT_EQ(*top_elem, expected);
  }
}

TEST_F(Test_Fixture_Feasible_Memory_Scheduler,
      schedule_all_possible_ops_ready_list) {

  dag_t::adjacency_map_t in = { {"A", {"B"}},  {"C", {"B"}},
    {"B", {}}, {"A_in", {"A"} }, {"B_in", {"B"} }, {"C_in", {"C"}} };
  dag_t::resource_cost_model_t memory = { {"A", 2UL}, {"A_in",2UL},
    {"C_in", 2}, {"C", 2}, {"B_in", 4UL}, {"B", 2UL} };
  dag_t::data_op_set_t data_ops = {"A_in", "B_in", "C_in"}; //data operations //

  dag_t g(in);
  g.reset_resource_model(memory);
  g.reset_data_op_set(data_ops);

  reset_input(g);
  init(resource_t(10));

  {
    op_list_t expected_ready_list = {"A", "C"};
    op_list_t sorted_ready_list = ready_list_;

    sorted_ready_list.sort();
    EXPECT_EQ(expected_ready_list, sorted_ready_list);
  }

  { 

    //current time is zero //
    EXPECT_EQ(schedule_all_possible_ready_ops(ready_list_.begin(),
            ready_list_.end()), 2UL);
    std::list<operation_t> scheduled_ops;
    get_currently_scheduled_operations(std::back_inserter(scheduled_ops));

    std::list<operation_t> expected_ops = {"A_in", "C_in"};
    scheduled_ops.sort();
    EXPECT_EQ(expected_ops, scheduled_ops);
  }

  {
    EXPECT_TRUE(is_operation_output_in_active_memory("A"));
    EXPECT_TRUE(is_operation_output_in_active_memory("C"));
  }
}

TEST_F(Test_Fixture_Feasible_Memory_Scheduler, test_auto_scheduled_data_ops) {

  dag_t::adjacency_map_t in = { {"A", {"B"}},  {"C", {"B"}},
    {"B", {}}, {"A_in", {"A"} }, {"B_in", {"B"} }, {"C_in", {"C"}} };
  dag_t::resource_cost_model_t memory = { {"A", 2UL}, {"A_in",2UL},
    {"C_in", 2}, {"C", 2}, {"B_in", 4UL}, {"B", 2UL} };
  dag_t::data_op_set_t data_ops = {"A_in", "B_in", "C_in"}; //data operations //

  dag_t g(in);
  g.reset_resource_model(memory);
  g.reset_data_op_set(data_ops);

  reset_input(g);
  init(resource_t(10));

  // schedule all possible ready_ops this should fire two compute ops//
  EXPECT_EQ( schedule_all_possible_ready_ops(ready_list_.begin(),
          ready_list_.end()), 2UL);

  {
    // since "A" and "C" are compute ops their data ops must be automatically 
    // scheduled.//

    std::list<operation_t> expected = { "A_in", "C_in" };
    std::list<operation_t> data_reads;

    get_currently_scheduled_operations(std::back_inserter(data_reads));
    data_reads.sort(); expected.sort();

    EXPECT_EQ(expected, data_reads);
  }

  {
    // now get the compute ops scheduled at time=1 //
    std::list<operation_t> expected = { "A", "C" };
    std::list<operation_t> compute_ops;


    get_scheduled_operations_at_time(schedule_time_t(1),
          std::back_inserter(compute_ops));

    compute_ops.sort(); expected.sort();
    EXPECT_EQ(expected, compute_ops);
  }
}
