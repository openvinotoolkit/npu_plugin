#include <gtest/gtest.h>
#include <vector>

#include "pass/lp_scheduler/lp_scheduler_pass.hpp"
#include "scheduler/scheduler_unit_test_utils.hpp"


typedef scheduler_unit_tests::Operation_Dag dag_t;
typedef mv::lp_scheduler::Repack_Input_DMA_Tasks<dag_t> repacker_t;

class Repack_Test_Fixture : public ::testing::Test {
  protected:
    typedef dag_t::adjacency_map_t adjacency_map_t;
    typedef dag_t::data_op_set_t data_op_set_t;
    typedef typename mv::lp_scheduler::scheduler_traits<dag_t> traits;
    typedef traits::scheduled_op_t scheduled_op_t;
    typedef traits::operation_t operation_t;
    typedef traits::data_op_selector_t data_op_selector_t;

    template<typename ScheduleIterator>
    scheduled_op_t get_scheduled_op(const operation_t& op, 
        ScheduleIterator sbegin, ScheduleIterator send) {
      for (; sbegin != send; ++sbegin) {
        if ((*sbegin).op_ == op) { return *sbegin; }
      }
      return scheduled_op_t();
    }

}; // class Repack_Test_Fixture // 

TEST_F(Repack_Test_Fixture, prefetch_limiting_consumer_test) {
  adjacency_map_t input= { {"Input", {"B", "Q", "P"}}, { "Q", {"T", "U"}},
    {"P", {"R", "S"}}, {"B_input", {"B"}}, {"P_input", {"P"}}, {"R", {}},
    {"S", {}}, {"T", {}}, {"U", {}}, {"UnusedMemory", {}} };
  data_op_set_t data_op_set = {"B_input", "P_input"};

  dag_t dag(input);
  dag.reset_data_op_set(data_op_set);

  // STEP-1: create some valid//
  std::vector<scheduled_op_t> original_schedule=  {
    {"Input", 1UL, 1UL, 5UL}, {"UnusedMemory", 1UL, 6UL, 100UL}, 
    {"Q", 3UL, 6UL, 10UL},
    {"P_input", 4UL, 25UL, 30UL}, // overlaps "UnusedMemory" //
    {"P", 5UL, 11UL, 15UL},
    {"R", 6UL, 50UL, 100UL},
    {"S", 7UL, 50UL, 100UL}, 
    {"T", 8UL, 50UL, 100UL},
    {"U", 9UL, 50UL, 100UL}, 
    {"B_input", 10UL, 6UL, 15UL}, // overlaps : P and Q //
    {"B", 11UL, 16UL, 20UL}
  };
  // P_input can be pre-fetched to time=1 since no address overlap //
  // B_input overlaps address range of P, Q but cannot be pre-fetched
  // until all consumers of P and Q are finished.
  data_op_selector_t data_op_selector(dag);

  repacker_t repacker(dag, data_op_selector);
  std::vector<scheduled_op_t> repacked_schedule;

  repacker.repack(original_schedule.begin(), original_schedule.end(), 
      std::back_inserter(repacked_schedule));

  EXPECT_EQ(repacked_schedule.size(), original_schedule.size());

  // P_input can be prefetched to time slot 3 instead of 4UL//
  scheduled_op_t p_input = get_scheduled_op("P_input",
        repacked_schedule.begin(), repacked_schedule.end());
  EXPECT_EQ(p_input.schedule_time_, 3UL);

  // B_input cannot be prefetched even if there is no address overlap //
  scheduled_op_t b_input = get_scheduled_op("B_input",
        repacked_schedule.begin(), repacked_schedule.end());
  EXPECT_EQ(b_input.schedule_time_, 10UL);

  // P should not be repacked //
  scheduled_op_t p_repacked = get_scheduled_op("P", repacked_schedule.begin(),
        repacked_schedule.end());
  scheduled_op_t p_original = get_scheduled_op("P", original_schedule.begin(),
        original_schedule.end());
  EXPECT_EQ(p_repacked.schedule_time_, p_original.schedule_time_);
}

TEST_F(Repack_Test_Fixture, linear_chain) {
  adjacency_map_t input= { {"Input", {"A"}}, { "A", {"B"}}, {"B", {"C"}},
    {"A_input", {"A"}}, {"B_input", {"B"}}, {"C_input", {"C"}}, {"C", {}},
    {"UnusedMemory", {}} };
  data_op_set_t data_op_set = {"A_input", "B_input", "C_input"};

  dag_t dag(input);
  dag.reset_data_op_set(data_op_set);

  // STEP-1: create some valid//
  std::vector<scheduled_op_t> original_schedule=  {
    {"UnusedMemory", 1UL, 1UL, 15UL}, {"Input", 1UL, 16UL, 95UL},
      {"A_input", 1UL, 96UL, 100UL}, 
    {"A", 2UL, 101UL, 105UL},
    {"B_input", 3UL, 1UL, 5UL}, 
    {"B", 4UL, 6UL, 10UL},
    {"C_input", 5UL, 11UL, 15UL},
    {"C", 6UL, 1UL, 5UL} };
    
  data_op_selector_t data_op_selector(dag);

  repacker_t repacker(dag, data_op_selector);
  std::vector<scheduled_op_t> repacked_schedule;

  repacker.repack(original_schedule.begin(), original_schedule.end(), 
      std::back_inserter(repacked_schedule));

  EXPECT_EQ(repacked_schedule.size(), original_schedule.size());

  // both B_input and C_input must be prefetched //
  scheduled_op_t p_input = get_scheduled_op("B_input",
        repacked_schedule.begin(), repacked_schedule.end());
  EXPECT_EQ(p_input.schedule_time_, 2UL);

  scheduled_op_t b_input = get_scheduled_op("C_input",
        repacked_schedule.begin(), repacked_schedule.end());
  EXPECT_EQ(b_input.schedule_time_, 2UL);
}
