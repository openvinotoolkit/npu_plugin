#include <vector>

#include "gtest/gtest.h"

#include "scheduler/scheduler_unit_test_utils.hpp"
#include "scheduler/dag_address_generator.hpp"


using namespace scheduler_unit_tests;
typedef Operation_Dag dag_t;

class DAG_Address_Generation_Test : public testing::Test {

  protected:

    ////////////////////////////////////////////////////////////////////////////
    typedef mv::lp_scheduler::Feasible_Memory_Schedule_Generator<dag_t>
        feasible_memory_scheduler_t;
    typedef feasible_memory_scheduler_t::resource_t resource_t;
    typedef feasible_memory_scheduler_t::operation_t operation_t;
    typedef size_t unit_t;

    
    struct select_all_nodes_t{
      bool operator()(const operation_t& op) const { return true; }
    }; // struct select_all_nodes_t //

    struct uniform_utility_t {
      size_t operator()(const operation_t& op) const { return 10UL; }
    }; // struct select_all_nodes_t //

    typedef mv::lp_scheduler::DAG_Address_Generator<dag_t, size_t,
              select_all_nodes_t, uniform_utility_t> dag_address_generator_t;
    typedef typename dag_address_generator_t::address_info_t address_info_t;

    struct scheduled_op_t {
      scheduled_op_t(size_t time, operation_t op) : time_(time), op_(op) {}
      size_t time_;
      operation_t op_;
    }; // struct scheduled_op_t //
    ////////////////////////////////////////////////////////////////////////////

    void generate_schedule(const dag_t& input, resource_t upper_bound) {
      feasible_memory_scheduler_t scheduler_begin(input, upper_bound),
                                  scheduler_end;
      schedule_.clear();
      for (; scheduler_begin != scheduler_end; ++scheduler_begin) {
        schedule_.push_back(
            scheduled_op_t((*scheduler_begin).time_,
                           (*scheduler_begin).op_));
      }
    }

    typedef std::unordered_map<operation_t, address_info_t> address_map_t;

    bool generate_addresses(const dag_t& input, unit_t upper_bound,
        address_map_t& output, size_t& high_watermark) {
      output.clear();
      dag_address_generator_t address_generator(input, upper_bound);
      std::vector<address_info_t> result;
      
      bool status = address_generator.generate(
          schedule_.begin(), schedule_.end(), std::back_inserter(result) );

      high_watermark = address_generator.get_high_watermark();

      for (auto ritr=result.begin(); ritr!=result.end(); ++ritr) {
        output[(*ritr).op_] = *ritr;
      }
      return status;
    }

    void SetUp() override {}
    void TearDown() override {}

    std::vector<scheduled_op_t> schedule_;
}; // class DAG_Address_Generator_Test //


TEST_F(DAG_Address_Generation_Test, diamond_test) {

  //////////////////////////////////////////////////////////////////////////////
  dag_t::adjacency_map_t in = { {"Input", {"A", "C"} },
      {"A", {"D"}}, {"C", {"D"}}, {"D", {}} };
  dag_t::resource_cost_model_t memory = { {"Input", 1UL}, {"A", 1UL},
    {"C", 1UL}, {"D", 0UL} };
  dag_t g(in);

  g.reset_resource_model(memory);
  //////////////////////////////////////////////////////////////////////////////


  // Generate a schedule (this is trivial topological ordering as the sum of
  // active resources of all the ops <= upper_bound).
  generate_schedule(g, resource_t(10UL));

  address_map_t address_map;
  size_t high_watermark = 0UL;

  generate_addresses(g, 100UL, address_map, high_watermark);

  // We select all nodes so we should have result equal to nodes //
  EXPECT_EQ(address_map.size(), 4UL);
  EXPECT_TRUE(high_watermark <= 30UL);

  // The addresses are going from low to high //
  EXPECT_EQ(address_map["Input"], address_info_t("Input", 1UL, 10UL) );

  // [1,10] is locked until A and C are completed//
  EXPECT_EQ(address_map["C"], address_info_t("C", 11UL, 20UL) );
  EXPECT_EQ(address_map["A"], address_info_t("A", 21UL, 30UL) );
}

TEST_F(DAG_Address_Generation_Test, linear_chain) {

  //////////////////////////////////////////////////////////////////////////////
  dag_t::adjacency_map_t in = { {"Input", {"A"} }, {"A", {"B"}},
      {"B", {"C"}}, {"C", {"D"}}, {"D", {"E"}}, {"E", {}} };
  dag_t::resource_cost_model_t memory = { {"Input", 1UL}, {"A", 1UL},
    {"B", 1UL}, {"C", 1UL}, {"D", 1UL}, {"E", 0UL} };
  dag_t g(in);

  g.reset_resource_model(memory);
  //////////////////////////////////////////////////////////////////////////////


  generate_schedule(g, resource_t(10UL));

  address_map_t address_map;
  size_t high_watermark = 0UL;

  // High water mark should not cross 20UL for a linear chain //
  generate_addresses(g, 100UL, address_map, high_watermark);

  // We select all nodes so we should have result equal to nodes //
  EXPECT_EQ(address_map.size(), 6UL);
  EXPECT_TRUE(high_watermark <= 20UL);

  // The addresses are going from low to high //
  EXPECT_EQ(address_map["Input"], address_info_t("Input", 1UL, 10UL) );

  EXPECT_EQ(address_map["A"], address_info_t("A", 11UL, 20UL) );
  EXPECT_EQ(address_map["B"], address_info_t("B", 1UL, 10UL) );
  EXPECT_EQ(address_map["C"], address_info_t("C", 11UL, 20UL) );
  EXPECT_EQ(address_map["D"], address_info_t("D", 1UL, 10UL) );
  EXPECT_EQ(address_map["E"], address_info_t("E", 11UL, 20UL) );
}

TEST_F(DAG_Address_Generation_Test, linear_chain_feed_forward) {

  //////////////////////////////////////////////////////////////////////////////
  dag_t::adjacency_map_t in = { {"Input", {"A", "F"} }, {"A", {"B"}},
      {"B", {"C"}}, {"C", {"D"}}, {"D", {"E"}}, {"E", {"F"}}, {"F", {}} };
  dag_t::resource_cost_model_t memory = { {"Input", 1UL}, {"A", 1UL},
    {"B", 1UL}, {"C", 1UL}, {"D", 1UL}, {"E", 1UL}, {"F", 0UL} };
  dag_t g(in);

  g.reset_resource_model(memory);
  //////////////////////////////////////////////////////////////////////////////


  generate_schedule(g, resource_t(10UL));

  address_map_t address_map;
  size_t high_watermark = 0UL;

  // High water mark should not cross 20UL for a linear chain //
  generate_addresses(g, 100UL, address_map, high_watermark);

  EXPECT_EQ(address_map.size(), 7UL);

  // Since address of Input are locked till end
  EXPECT_TRUE(high_watermark <= 30UL); 

  // [1, 10] is locked until all the ops are finished //
  EXPECT_EQ(address_map["Input"], address_info_t("Input", 1UL, 10UL) );
  EXPECT_EQ(address_map["A"], address_info_t("A", 11UL, 20UL) );
  EXPECT_EQ(address_map["B"], address_info_t("B", 21UL, 30UL) );
  EXPECT_EQ(address_map["C"], address_info_t("C", 11UL, 20UL) );
  EXPECT_EQ(address_map["D"], address_info_t("D", 21UL, 30UL) );
  EXPECT_EQ(address_map["E"], address_info_t("E", 11UL, 20UL) );
  EXPECT_EQ(address_map["F"], address_info_t("F", 21UL, 30UL) );
}

TEST_F(DAG_Address_Generation_Test, fully_active_external_nodes_in_tree) {

  //////////////////////////////////////////////////////////////////////////////
  dag_t::adjacency_map_t in = { {"Input", {"A", "B"} }, {"A", {"C", "D"}},
    {"B", {"E", "F"}}, {"C", {"G"}}, {"D", {"G"}}, {"E", {"G"}}, {"F", {"G"}},
    {"G", {}} };
  dag_t::resource_cost_model_t memory = { {"Input", 1UL}, {"A", 1UL},
    {"B", 1UL}, {"C", 1UL}, {"D", 1UL}, {"E", 1UL}, {"F", 1UL}, {"G", 0UL} };
  dag_t g(in);

  g.reset_resource_model(memory);
  //////////////////////////////////////////////////////////////////////////////


  // increase the active resources so the scheduler will schedule all the
  // children (C,D,E,F} at the same time //
  generate_schedule(g, resource_t(50UL));

  address_map_t address_map;
  size_t high_watermark = 0UL;

  // High water mark should not cross 20UL for a linear chain //
  generate_addresses(g, 100UL, address_map, high_watermark);

  EXPECT_EQ(address_map.size(), 8UL);

  // Since address of Input are locked till end
  EXPECT_TRUE(high_watermark <= 60UL); 

  // Address for the children must be mutually disjoint //
  std::vector<std::string> child_nodes = {"C", "D", "E", "F"};

  size_t count = 0;
  for (size_t i=0; i<child_nodes.size(); i++) {
    for (size_t j=i+1; j<child_nodes.size(); j++) {
      EXPECT_TRUE(
          address_map[child_nodes[i]].is_disjoint(
              address_map[child_nodes[j]] ) );
      ++count;
    }
  }
  EXPECT_EQ(count, 6UL /* 4 choose 2 */ );
}
