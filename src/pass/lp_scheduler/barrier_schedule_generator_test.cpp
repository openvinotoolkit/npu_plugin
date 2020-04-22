#include "gtest/gtest.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pass/lp_scheduler/barrier_schedule_generator.hpp>
#include <pass/lp_scheduler/barrier_simulator.hpp>
#include <scheduler/scheduler_unit_test_utils.hpp>

typedef mv::lp_scheduler::Barrier_Resource_State barrier_state_t;
typedef typename barrier_state_t::barrier_t barrier_t;

TEST(Barrier_Resource_State, zero_one_barriers) {
  barrier_state_t state;

  state.init(10, 1);

  EXPECT_FALSE(state.has_barrier_with_slots(2UL));

  for (size_t b=1UL; b<=10UL; b++) {
    EXPECT_TRUE(state.has_barrier_with_slots(1UL));
    barrier_t assigned_barrier = state.assign_slots(1UL);
    EXPECT_EQ(assigned_barrier, barrier_t(b));
  }
}

TEST(Barrier_Resource_State, zero_one_barriers_dynamic) {
  barrier_state_t state;

  size_t n = 10UL, m = 1UL;
  state.init(n, m);

  // assign all resources //
  for (size_t b=1UL; b<=n; b++) {
    state.assign_slots(1UL);
  }


  // Invariant //
  EXPECT_FALSE(state.has_barrier_with_slots(1UL));
  for (size_t b=n; b>=1UL; b--) {
    // if you free up barrier b then it should be available immediately //

    EXPECT_FALSE(state.has_barrier_with_slots(1UL));

    // 2. Free update barrier b //
    EXPECT_TRUE(state.unassign_slots(b, 1UL));

    // 3. Make sure we can claim the same barrier again //
    barrier_t assigned_barrier = state.assign_slots(1UL);
    EXPECT_EQ(assigned_barrier, barrier_t(b));
  }
  // All barriers should still be filled //
  EXPECT_FALSE(state.has_barrier_with_slots(1UL));
}

TEST(Barrier_Resource_State,  non_unit_slots_round_robin){
  barrier_state_t state;

  size_t n = 10UL, m = 5UL;
  state.init(n, m);

  EXPECT_FALSE(state.has_barrier_with_slots(7UL));
  std::vector<size_t> partition = {2UL, 1UL, 2UL};

  // test partition into 2+1+2 //

  // total slots = 50 so we should satisfy the demand of 10 partition calls//
  // as follows.

  // ROUND-1: 
  // free slots (before): 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 
  // barrier ids        : 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 
  // free slots (after) : 3, 4, 3, 3, 4, 3, 3, 4, 3, 3  

  // ROUND-2:
  // free slots (before) : 3, 4, 3, 3, 4, 3, 3, 4, 3, 3  
  // barrier ids         : 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 
  // free slots (after)  : 2, 2, 1, 2, 2, 1, 2, 2, 1, 2
  for (size_t call=1UL; call<=n; call++) {

    for (size_t p=0; p<partition.size(); ++p) {
      EXPECT_TRUE(state.has_barrier_with_slots(partition[p]));
      barrier_t assigned_barrier = state.assign_slots(partition[p]);
      ASSERT_TRUE(assigned_barrier <= 10UL);
    }

  }



  EXPECT_FALSE(state.has_barrier_with_slots(1UL));
}

TEST(Barrier_Resource_State,  non_unit_slots_dynamic){
  barrier_state_t state;

  size_t n = 10UL, m = 5UL;
  state.init(n, m);

  // assign all resources //
  for (size_t b=1UL; b<=n; b++) {
    state.assign_slots(m);
  }

  EXPECT_FALSE(state.has_barrier_with_slots(1UL));
  std::vector<size_t> partition = {2UL, 1UL, 2UL};

  // test partition into 2+1+2 //
  for (size_t b=n; b>=1UL; b--) {
    // free update barrier b //

    EXPECT_FALSE(state.has_barrier_with_slots(m));
    EXPECT_TRUE(state.unassign_slots(b, m));

    for (size_t p=0; p<partition.size(); ++p) {
      EXPECT_TRUE(state.has_barrier_with_slots(partition[p]));
      barrier_t assigned_barrier = state.assign_slots(partition[p]);
      EXPECT_EQ(assigned_barrier, barrier_t(b));
    }
  }

  EXPECT_FALSE(state.has_barrier_with_slots(1UL));
}

////////////////////////////////////////////////////////////////////////////////
// Barrier scheduler unit-tests //
class Test_Fixture_Barrier_Scheduler : public testing::Test {

  protected:
  //////////////////////////////////////////////////////////////////////////////
  typedef scheduler_unit_tests::Operation_Dag dag_t;
  typedef mv::lp_scheduler::Barrier_Schedule_Generator<dag_t>
      barrier_scheduler_t;
  typedef barrier_scheduler_t::schedule_info_t schedule_info_t;
  typedef barrier_scheduler_t::schedule_time_t schedule_time_t;
  typedef typename dag_t::adjacency_map_t adjacency_map_t;
  typedef typename dag_t::operation_t operation_t;
  typedef typename dag_t::adjacency_list_t adjacency_list_t;
  struct schedule_results_t {
    schedule_results_t(size_t sops=0UL, size_t mactiv=0UL, size_t mspan=0UL) :
      scheduled_ops_(sops), max_active_barriers_(mactiv), make_span_(mspan) {}

    bool operator==(const schedule_results_t& o) const {
      return (scheduled_ops_ == o.scheduled_ops_) &&
        (max_active_barriers_ == o.max_active_barriers_) &&
        (make_span_ == o.make_span_);
    }


    size_t scheduled_ops_;
    size_t max_active_barriers_;
    size_t make_span_;
  }; // struct schedule_results_t //
  //////////////////////////////////////////////////////////////////////////////

  void create_degree_bounded_tree_test(dag_t& input,
        size_t degree, size_t depth) {
    char buf[8192]; // size_t cant exceed 8192 digits//
    size_t nid=0;
    std::vector<size_t> level_list[2];
    size_t level = 0UL;
    level_list[0].push_back(nid++);
    adjacency_map_t tree;

    while (level < depth) {
      std::vector<size_t> &curr_level = level_list[level%2UL];
      std::vector<size_t> &next_level = level_list[(level+1UL)%2UL];

      next_level.clear();
      for (auto citr=curr_level.begin(); citr!=curr_level.end(); ++citr) {
        // create new node//
        sprintf(buf, "N-%lu", *citr);
        typename adjacency_map_t::iterator nitr =
            (tree.insert(std::make_pair(buf, adjacency_list_t()))).first;
        // now add d edges //
        for (size_t d=0; d<degree; ++d) {
          size_t curr_node = nid++;
          next_level.push_back(curr_node);
          sprintf(buf, "N-%lu", curr_node);
          (nitr->second).push_back(buf);
        }
      }
      curr_level.clear();
      ++level;
    }

    // add the leaf nodes //
    std::vector<size_t> *leaf_nodes = level_list[0UL].empty() ?
        &(level_list[1UL]) : &(level_list[0UL]);
    for (auto litr=leaf_nodes->begin(); litr!=leaf_nodes->end(); ++litr) {
      sprintf(buf, "N-%lu", *litr);
      tree.insert(std::make_pair(buf, adjacency_list_t()));
    }

    input.reset(tree);
  }

  // Runs the scheduler and computes the schedule_results_t //
  schedule_results_t
    run_scheduler_and_get_max_active_barriers_and_scheduled_ops(
        const dag_t& in, size_t bcount, size_t slot_count, bool dump_info=false)
  {
    std::unordered_map<schedule_time_t, std::set<size_t> >
        active_barriers_at_time;
    std::unordered_set<operation_t> scheduled_ops;
    schedule_results_t result;
    size_t make_span = 0UL;
    
    barrier_scheduler_t bscheduler(in, bcount, slot_count), bscheduler_end;
    for (;bscheduler != bscheduler_end; ++bscheduler) {
      const schedule_info_t& sinfo = *bscheduler;

      if (dump_info) {
        printf("[time=%lu barrier_id=%lu slot_count=%lu op=%s]\n",
            sinfo.schedule_time_, sinfo.barrier_index_, sinfo.slot_count_,
            sinfo.op_.c_str());
      }

      scheduled_ops.insert(sinfo.op_);
      if (active_barriers_at_time.find(sinfo.schedule_time_) ==
          active_barriers_at_time.end()) {
        active_barriers_at_time[sinfo.schedule_time_] = std::set<size_t>();
      }
      active_barriers_at_time[sinfo.schedule_time_].insert(
            sinfo.barrier_index_);
      scheduled_ops.insert(sinfo.op_);
      make_span = std::max(make_span, sinfo.schedule_time_);
    }

    size_t max_active = 0UL;
    for (auto aitr=active_barriers_at_time.begin();
        aitr!=active_barriers_at_time.end(); ++aitr) {
      max_active = std::max(max_active, (aitr->second).size());
    }

    result.scheduled_ops_ = scheduled_ops.size();
    result.max_active_barriers_ = max_active;
    result.make_span_ = make_span;
    return result;
  }

}; // class Test_Fixture_Barrier_Scheduler //


TEST_F(Test_Fixture_Barrier_Scheduler, barriers_2_slots_1_depth_1_degree_100) {
  dag_t input_dag;

  //Input: depth=1UL degree=100UL //
  create_degree_bounded_tree_test(input_dag, 100UL, 1UL);

  EXPECT_EQ(input_dag.size(), 101UL);
  // create a 1 level tree with 100 nodes //
  schedule_results_t results =
    run_scheduler_and_get_max_active_barriers_and_scheduled_ops(input_dag,
        2UL, 1UL);
  EXPECT_EQ(results, schedule_results_t(101UL, 2UL, 50UL) );
}

TEST_F(Test_Fixture_Barrier_Scheduler, barriers_2_slots_1_depth_100_degree_1) {
  dag_t input_dag;

  // this is a chain of length 100//
  create_degree_bounded_tree_test(input_dag, 1UL, 100UL);

  EXPECT_EQ(input_dag.size(), 101UL);
  // create a 1 level tree with 100 nodes //
  schedule_results_t results =
    run_scheduler_and_get_max_active_barriers_and_scheduled_ops(input_dag,
        2UL, 1UL);
  // the dependencies force us to use only one barrier at max //
  EXPECT_EQ(results, schedule_results_t(101UL, 1UL, 100UL) );
}

TEST_F(Test_Fixture_Barrier_Scheduler,
      barriers_100_slots_1_depth_1_degree_100) {
  dag_t input_dag;

  // Input: is a chain of length 100//
  create_degree_bounded_tree_test(input_dag, 100UL, 1UL);

  EXPECT_EQ(input_dag.size(), 101UL);

  // schedule with 100 barriers //
  schedule_results_t results =
    run_scheduler_and_get_max_active_barriers_and_scheduled_ops(input_dag,
        100UL, 1UL);
  // the dependencies force us to use only one barrier at max //
  EXPECT_EQ(results, schedule_results_t(101UL, 100UL, 1UL) );
}

TEST_F(Test_Fixture_Barrier_Scheduler,
      barriers_2_slots_1_depth_3_degree_2) {
  dag_t input_dag;

  // Input: binary tree of depth 3 //
  create_degree_bounded_tree_test(input_dag, 2UL, 3UL);

  EXPECT_EQ(input_dag.size(), 15UL);

  // schedule with 2 barriers //
  schedule_results_t results =
    run_scheduler_and_get_max_active_barriers_and_scheduled_ops(input_dag,
        2UL, 1UL);
  EXPECT_EQ(results, schedule_results_t(15UL, 2UL, 7UL) );
}

TEST_F(Test_Fixture_Barrier_Scheduler,
      barriers_8_slots_1_depth_3_degree_2) {
  dag_t input_dag;

  // Input: binary tree of depth 3 //
  create_degree_bounded_tree_test(input_dag, 2UL, 3UL);

  EXPECT_EQ(input_dag.size(), 15UL);

  // schedule with 2 barriers //
  schedule_results_t results =
    run_scheduler_and_get_max_active_barriers_and_scheduled_ops(input_dag,
        8UL, 1UL);
  // makespan will be log(n) depth //
  EXPECT_EQ(results, schedule_results_t(15UL, 8UL, 3UL) );
}


TEST_F(Test_Fixture_Barrier_Scheduler, barrier_1_slots_100_depth_1_degree_100) {
  dag_t input_dag;

  create_degree_bounded_tree_test(input_dag, 100UL, 1UL);
  schedule_results_t results =
    run_scheduler_and_get_max_active_barriers_and_scheduled_ops(input_dag,
        1UL, 100UL);
  EXPECT_EQ(results, schedule_results_t(101UL, 1UL, 1UL) );
}


////////////////////////////////////////////////////////////////////////////////
// Barrier simulator unit-tests //
class Test_Fixture_Barrier_Simulation_Checker : public testing::Test {

  protected:
  //////////////////////////////////////////////////////////////////////////////
  typedef scheduler_unit_tests::Operation_Dag dag_t;
  typedef typename dag_t::operation_t operation_t;

  struct op_selector_t {
    static bool is_barrier_op(const dag_t& in, const operation_t& op) {
      const char * br_str= "Barrier-";
      const char * op_str = op.c_str();

      while ((*op_str && *br_str) && (*op_str == *br_str)) {
        ++op_str; ++br_str;
      }
      return *br_str == '\0';
    }

    static bool is_compute_op(const dag_t& in, const operation_t& op) {
      return !is_barrier_op(in, op);
    }

    static bool is_data_op(const dag_t& in, const operation_t& op) {
      return false;
    }
  }; // struct op_selector_t //

  struct real_barrier_mapper_t {
    size_t operator()(const dag_t& in, const operation_t& op) const {
      size_t real_barrier_index;
      int ret = sscanf(op.c_str(), "%*[^-]%*c%lu", &real_barrier_index);
      assert(ret == 1);
      return real_barrier_index;
    }
  }; // struct real_barrier_mapper_t //


  typedef mv::lp_scheduler::Runtime_Barrier_Simulation_Checker<dag_t,
          op_selector_t, real_barrier_mapper_t> runtime_checker_t;

}; // class Test_Fixture_Barrier_Simulation_Checker //


TEST_F(Test_Fixture_Barrier_Simulation_Checker, parallel_path_clogged) {
  dag_t::adjacency_map_t in = { {"start", {"Barrier-0", "Barrier-1"}},
    // parallel path 1 (even path)//
      // tasks in this path //
    {"task-0", {"Barrier-2"}}, {"task-2", {"Barrier-4"}}, {"task-4", {} }, 
      // barriers in this path //
    {"Barrier-0", {"task-0", "task-5"} }, 
    {"Barrier-2", {"task-2", "task-5"} },
    {"Barrier-4", {"task-4"} },

    // parallel path 2 (odd path)//
      // tasks in this path //
    {"task-1", {"Barrier-3"}}, {"task-3", {"Barrier-5"}}, {"task-5", {} }, 
      // barriers in this path //
    {"Barrier-1", {"task-1", "task-4"} }, 
    {"Barrier-3", {"task-3", "task-4"} },
    {"Barrier-5", {"task-5"} }
  };


  dag_t input(in);
  size_t barrier_bound = 2UL;
  runtime_checker_t checker(input, barrier_bound);
  EXPECT_FALSE(checker.check());
}


TEST_F(Test_Fixture_Barrier_Simulation_Checker, parallel_path_unclogged) {
  dag_t::adjacency_map_t in = { {"start", {"Barrier-0", "Barrier-1"}},
    // parallel path 1 (even path)//
      // tasks in this path //
    {"task-0", {"Barrier-2"}}, {"task-2", {"Barrier-4"}}, {"task-4", {} }, 
      // barriers in this path //
    {"Barrier-0", {"task-0"} }, 
    {"Barrier-2", {"task-2", "task-5"} },
    {"Barrier-4", {"task-4"} },

    // parallel path 2 (odd path)//
      // tasks in this path //
    {"task-1", {"Barrier-3"}}, {"task-3", {"Barrier-5"}}, {"task-5", {} }, 
      // barriers in this path //
    {"Barrier-1", {"task-1"}  }, 
    {"Barrier-3", {"task-3", "task-4"} },
    {"Barrier-5", {"task-5"} }
  };


  dag_t input(in);
  size_t barrier_bound = 2UL;
  runtime_checker_t checker(input, barrier_bound);
  EXPECT_TRUE(checker.check());
}
