#include <stdlib.h>

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "pass/lp_scheduler/operation_precedence_dag.hpp"

typedef mv::scheduler::Operation_Dag<> op_model_dag_t;


class Operation_Dag_Test : public ::testing::Test {
  protected:

    void SetUp() override {
      setup_three_layer_conv_input();
    }

    Operation_Dag_Test() : three_layer_conv_input_("3layer_conv") {}

    mv::OpModel& three_layer_conv_model() {
      return three_layer_conv_input_.model();
    }

  private:


    //TODO(vamsikku): remove the external reference to the .json compilation
    //descriptor instead just create a default JSON structure.
    void setup_three_layer_conv_input() {
      double inf = std::numeric_limits<double>::infinity();

      std::string prefix(getenv("MCM_HOME"));

      mv::CompilationUnit &unit = three_layer_conv_input_;
      mv::OpModel& om = unit.model();
      auto input0 = om.input({56,56,3,1}, mv::DType("UInt8"),
          mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}},
            "input#9");

      std::vector<int64_t> weightsData0 =
          mv::utils::generateSequence<int64_t> (3*3*3*64);
      auto weights0 = om.constantInt(weightsData0,{3,3,3,64},
            mv::DType("UInt8"), mv::Order::getZMajorID(4),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}}, "conv#0_weights#1");

      auto conv0 = om.conv(input0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#10");

      std::vector<int64_t> biasWeightsData0 =
          mv::utils::generateSequence<int64_t> (64);
      auto biasWeights0 = om.constantInt(biasWeightsData0,{64},
          mv::DType("UInt8"), mv::Order::getColMajorID(1),
          {{0},{1.9952236470999196e-05},{-inf},{inf}}, "conv#0_bias#2");
      auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"),
          {{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> weightsData1 =
          mv::utils::generateSequence<int64_t> (3*3*64*128);
      auto weights1 = om.constantInt(weightsData1,{3,3,64,128},
          mv::DType("UInt8"), mv::Order::getZMajorID(4),
          {{125},{0.003295167814940214},{-0.41293057799339294},
            {0.4273372292518616}}, "conv_1#3_weights#4");
      auto conv1 = om.conv(bias_c0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv_1#11");

      std::vector<int64_t> biasWeightsData1 =
          mv::utils::generateSequence<int64_t> (128);
      auto biasWeights1 = om.constantInt(biasWeightsData1,{128},
          mv::DType("UInt8"), mv::Order::getColMajorID(1),
          {{0},{1.292222714255331e-05},{-inf},{inf}}, "conv_1#3_bias#5");
      auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"),
          {{0},{0.003921568859368563},{0.0},{1.0}});

      std::vector<int64_t> weightsData2 =
        mv::utils::generateSequence<int64_t> (3*3*128*128);
      auto weights2 = om.constantInt(weightsData2,{3,3,128,128},
          mv::DType("UInt8"), mv::Order::getZMajorID(4),
          {{118},{0.0037134578451514244},{-0.44002026319503784},
            {0.5069115161895752}}, "output#6_weights#7");
      auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
          "output#12");

      std::vector<int64_t> biasWeightsData2 =
          mv::utils::generateSequence<int64_t> (128);
      auto biasWeights2 = om.constantInt(biasWeightsData2,{128},
          mv::DType("UInt8"), mv::Order::getColMajorID(1),
            {{0},{1.4562579963239841e-05},{-inf},{inf}}, "output#6_bias#8");
      auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"),
            {{0},{0.003921568859368563},{0.0},{1.0}});

      om.output(bias_c2);

      //TODO(vamsikku): REMOVE THIS PATH //
      std::string compDescPath =
        prefix + "/tests/system_tests/lp_scheduler/"
        "three_layer_conv_model/comp_desc_unit_test.json";
        
      unit.loadCompilationDescriptor(compDescPath);

      unit.loadTargetDescriptor(mv::Target::ma2490);
      unit.initialize();
      unit.run();
    }

    mv::CompilationUnit three_layer_conv_input_;
}; // class Operation_Dag_Test //


typedef mv::scheduler::Operation_Dag<mv::ControlModel> control_dag_t;
TEST_F(Operation_Dag_Test, test_structure_data_dag) {
  mv::ControlModel cm(three_layer_conv_model());
  control_dag_t input(cm);

  auto itr = input.begin_nodes();
  auto itr_end = input.end_nodes();

  size_t conv_op_count = 0UL, input_count = 0UL, output_count=0UL,
         total_op_count = 0UL; 

  while (itr != itr_end) {
    const op_model_dag_t::operation_t& op = *itr;
    if (op->getOpType() == "DPUTask") { conv_op_count++; }
    else if (op->getOpType() == "Input") { input_count++; }
    else if (op->getOpType() == "Output") { output_count++; }

    ++total_op_count;
    ASSERT_TRUE(total_op_count <= 100UL);
    ++itr;
  }

  EXPECT_EQ(conv_op_count, 4UL);
  EXPECT_EQ(input_count, 1UL);
  EXPECT_EQ(output_count, 1UL);
}

TEST_F(Operation_Dag_Test, in_degree_test) {
  typedef std::unordered_map<op_model_dag_t::operation_t, size_t> indegree_map_t;
  indegree_map_t indegree;

  op_model_dag_t dag( three_layer_conv_model() );
  size_t total_iteration_count = 0UL; // should not exceed the edges 13UL //
  for (op_model_dag_t::const_operation_iterator_t itr = dag.begin_nodes();
      itr != dag.end_nodes(); ++itr) {

    // scan the outgoing edges and add the indegree of all outgoing nodes. //
    for (op_model_dag_t::const_operation_iterator_t jtr = dag.begin_nodes(*itr);
          jtr != dag.end_nodes(*itr); ++jtr) {
      indegree_map_t::iterator iitr = indegree.find(*jtr);
      if (iitr == indegree.end()) {
        iitr = (indegree.insert(std::make_pair(*jtr, 0UL))).first;
      }
      iitr->second++;
      ASSERT_TRUE(total_iteration_count <= 13UL);
    }
    ASSERT_TRUE(total_iteration_count <= 13UL);
  }

  for (indegree_map_t::const_iterator itr=indegree.begin(); itr!=indegree.end();
      ++itr) {
    std::string op_type = (itr->first)->getOpType();
    ASSERT_TRUE((op_type != "DPUTask") ||
        ((itr->second == 4) || (itr->second == 3)));
    ASSERT_TRUE((op_type != "DMATask") || (itr->second == 1));
  }
}

TEST_F(Operation_Dag_Test, mv_lp_scheduler_basic_test) {
  op_model_dag_t dag( three_layer_conv_model() );
  size_t max_memory = 2000000;
  mv::scheduler::mv_lp_scheduler_t scheduler_begin(dag, max_memory),
      scheduler_end;

  std::unordered_map<size_t, size_t> memory_use_map;

  while (scheduler_begin != scheduler_end) {
    auto rstate = scheduler_begin.resource_state();
    auto rinfo = rstate.get_resource_usage_info(*scheduler_begin);
    auto mitr = memory_use_map.find(scheduler_begin.current_time());
    auto curr_time = scheduler_begin.current_time();

    if (mitr == memory_use_map.end()) {
      memory_use_map[curr_time] = 0UL;
    }
    memory_use_map[curr_time] += ((rinfo.end_ - rinfo.begin_)+1);

    std::cout << "op=" << (*scheduler_begin)->getOpType() << " time="
      << scheduler_begin.current_time() << " mem=[" << rinfo.begin_ << " " << 
        rinfo.end_ << "] " << std::endl;

    ++scheduler_begin;
  }
  // make sure memory is within limit at all times //

  for (auto mitr=memory_use_map.begin(); mitr!=memory_use_map.end(); ++mitr) {
    ASSERT_TRUE( mitr->second <= max_memory);
  }
}

TEST_F(Operation_Dag_Test, incoming_edge_iterator) {
  op_model_dag_t dag( three_layer_conv_model() );
  typename op_model_dag_t::operation_t op = dag.get_op_by_name("conv_1#11");
  ASSERT_TRUE(op != NULL);

  std::unordered_set<std::string> expected_parents = {
      "conv_1#3_weights#4_DDR2CMX",  "conv_1#11_weights_table_DDR2CMX",
      "conv#10" };
  typename op_model_dag_t::const_operation_iterator_t itr =
      dag.begin_parent_nodes(op), itr_end = dag.end_parent_nodes(op);

  ASSERT_TRUE(itr != itr_end);

  std::unordered_set<std::string> found_parents;
  for (size_t i=0; itr!=itr_end; ++itr,++i) {
    ASSERT_TRUE(i < expected_parents.size());
    found_parents.insert((*itr)->getName());
  }
  EXPECT_EQ(found_parents, expected_parents);

  op = dag.get_op_by_name("conv#10_sparse_dw_DDR2CMX");
  ASSERT_TRUE(op != NULL);
  itr = dag.begin_parent_nodes(op);
  itr_end = dag.end_parent_nodes(op);
  EXPECT_EQ(itr, itr_end);
}

























