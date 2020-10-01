#include <stdlib.h>

#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/target/kmb/barrier_deps.hpp"
#include "pass/lp_scheduler/operation_precedence_dag.hpp"
#include "pass/lp_scheduler/lp_scheduler_pass.hpp"
#include "pass/lp_scheduler/barrier_scheduler_pass.hpp"

typedef mv::scheduler::Operation_Dag<> op_model_dag_t;

class Operation_Dag_Test : public ::testing::Test {
  protected:

    void SetUp() override {
      //TODO(vamsikku): REMOVE THIS PATH DEPENDENCY. Create a JSON file
      //programatically//
      std::string prefix(getenv("MCM_HOME"));
      comp_desc_path_ = prefix + 
          "/config/compilation/release_kmb-sc.json";
      setup_three_layer_conv_input();
    }

    Operation_Dag_Test() : three_layer_conv_input_("3layer_conv"),
      upa_chain_ending_with_dpu_("upa_chain_ending_with_dpu"),
      upa_chain_with_single_link_("upa_chain_with_single_link_"),
      upa_chain_with_multiple_link_("upa_chain_with_multiple_link_") {}

    mv::OpModel& three_layer_conv_model() {
      return three_layer_conv_input_.model();
    }


    mv::OpModel& clear_control_model_and_remove_barriers(mv::OpModel& om) {
      // clear all control edges //
      mv::ControlModel cm(om);

      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd();) {
        fitr_next = fitr; ++fitr_next;
        cm.undefineFlow(fitr);
        fitr = fitr_next;
      }

      // remove all barrier ops //
      std::vector<mv::Data::OpListIterator> ops_to_remove;
      for (mv::Data::OpListIterator oitr=om.getInput(); oitr!=om.opEnd();
            ++oitr) {
        if (!(oitr->getOpType() == "BarrierTask")) { continue; }

        ops_to_remove.push_back(oitr);
      }

      for (auto ritr=ops_to_remove.begin(); ritr!=ops_to_remove.end(); ++ritr) {
        om.removeOp(*ritr);
      }

      // add control edges equal to the edges in the op model //
      for (mv::Data::OpListIterator oitr=om.getInput(); oitr!=om.opEnd();
          ++oitr) {
        mv::Data::OpListIterator source = oitr; 
        if (source->isImplicit()) {
          for (mv::Data::OpParentIterator pitr=oitr.leftmostParent();
               pitr!=om.opEnd(); ++pitr) {
            for (mv::Data::OpChildIterator citr=oitr.leftmostChild();
                  citr!=om.opEnd(); ++citr) {
              mv::Data::OpListIterator parent = om.getOp(pitr->getName());
              mv::Data::OpListIterator child = om.getOp(citr->getName());
              assert(parent != om.opEnd());
              assert(child != om.opEnd());
              cm.defineFlow(parent, child);
            }
          }
        } else {
          for (mv::Data::OpChildIterator citr=oitr.leftmostChild();
                citr!=om.opEnd(); ++citr) {
            mv::Data::OpListIterator sink = om.getOp(citr->getName());
            assert(sink != om.opEnd());
            cm.defineFlow(source, sink);
          }
        }
      }
      return om;
    }

    template<typename op_itr>
    bool is_barrier_op(op_itr itr) const {
      return (itr->getOpType() == "BarrierTask");
    }

    bool has_valid_barrier_in_and_out_references(mv::ControlModel& cm, 
        mv::Data::OpListIterator op) const {
      mv::Control::OpListIterator cop_itr = cm.switchContext(op);
      if (cop_itr == cm.opEnd()) { return true; }

      // check the parents //
      for (mv::Control::OpParentIterator pitr=cop_itr.leftmostParent();
          pitr!=cm.opEnd(); ++pitr) {

        if (cop_itr->getOpType() == "Output") { continue; }
        if (is_barrier_op(cop_itr) == is_barrier_op(pitr)) { return false; }

        if (is_barrier_op(cop_itr)) {
          const mv::Barrier& barrier = cop_itr->get<mv::Barrier>("Barrier");
          const mv::BarrierDependencies& deps =
              pitr->get<mv::BarrierDependencies>("BarrierDeps");
          if (!deps.hasUpdateBarrierWithID(barrier.getID())) { return false; }
        } else {
          const mv::Barrier& barrier = pitr->get<mv::Barrier>("Barrier");
          const mv::BarrierDependencies& deps =
              cop_itr->get<mv::BarrierDependencies>("BarrierDeps");
          if (!deps.hasWaitBarrierWithID(barrier.getID())) { return false; }
        }
      }

      // check the children //
      for (mv::Control::OpChildIterator citr=cop_itr.leftmostChild();
          citr!=cm.opEnd(); ++citr) {

        if (citr->getOpType() == "Output") { continue; }
        if (is_barrier_op(cop_itr) == is_barrier_op(citr)) { return false; }

        if (is_barrier_op(cop_itr)) {
          const mv::Barrier& barrier = cop_itr->get<mv::Barrier>("Barrier");
          const mv::BarrierDependencies& deps =
              citr->get<mv::BarrierDependencies>("BarrierDeps");
          if (!deps.hasWaitBarrierWithID(barrier.getID())) { return false; }
        } else {
          const mv::Barrier& barrier = citr->get<mv::Barrier>("Barrier");
          const mv::BarrierDependencies& deps =
              cop_itr->get<mv::BarrierDependencies>("BarrierDeps");
          if (!deps.hasUpdateBarrierWithID(barrier.getID())) { return false; }
        }
      }
      return true;
    }


    // Checks if the control edges exists between 
    bool is_valid_barrier_schedule(mv::ControlModel& cm) const {
      mv::OpModel om(cm);
      for (mv::Data::OpListIterator oitr=om.getInput(); oitr!=om.opEnd();
            ++oitr) {
        if (!has_valid_barrier_in_and_out_references(cm, oitr)) {
          return false;
        }
      }
      return true;
    }

    // barrier_ids must be in range [0...)
    bool does_barrier_ids_have_no_gaps(mv::ControlModel& cm) {
      mv::OpModel om(cm);
      std::vector<size_t> barrier_ids;
      for (mv::Data::OpListIterator oitr=om.getInput(); oitr!=om.opEnd();
            ++oitr) {
        if (!(oitr->getOpType() == "BarrierTask")) { continue; }
        mv::Barrier &barrier = oitr->get<mv::Barrier>("Barrier");
        barrier_ids.push_back(barrier.getID());
      }

      std::sort(barrier_ids.begin(), barrier_ids.end());
      for (size_t i=0; i<barrier_ids.size(); i++) {
        if (barrier_ids[i] != i) { return false; }
      }
      return true;
    }



    //TODO(vamsikku): remove the external reference to the .json compilation
    //descriptor instead just create a default JSON structure.
    void setup_three_layer_conv_input() {
      double inf = std::numeric_limits<double>::infinity();


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

      std::string compDescPath(comp_desc_path_);
      unit.loadCompilationDescriptor(compDescPath);

      unit.loadTargetDescriptor(mv::Target::ma2490);
      unit.initialize();
      unit.run();
    }


    mv::OpModel& upa_chain_ending_with_dpu_model() {
      return upa_chain_ending_with_dpu_.model();
    }

    void setup_upa_chain_ending_with_dpu_model() {
      mv::CompilationUnit &unit = upa_chain_ending_with_dpu_;
      mv::OpModel& om = unit.model();
      
      auto input0 = om.input({1,1,1000,1}, mv::DType("Float64"),
          mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0#4");
      std::string axis = "C";
     
      auto softmax0 = om.softmax(input0, axis);
      auto softmax1 = om.softmax(softmax0, axis);
      auto softmax2 = om.softmax(softmax1, axis, mv::DType("Float64"),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}});

      std::vector<int64_t> weightsData0 =
          mv::utils::generateSequence<int64_t> (1*1*1000*1);
      auto weights0 = om.constantInt(weightsData0,{1,1,1000,1},
            mv::DType("UInt8"), mv::Order::getZMajorID(4),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}}, "conv#0_weights#1");

      auto conv0 = om.conv(softmax2, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#10");

      om.output(conv0);

      std::string compDescPath(comp_desc_path_);
      unit.loadCompilationDescriptor(compDescPath);

      unit.loadTargetDescriptor(mv::Target::ma2490);
      unit.initialize();
      unit.run();
    }


    void setup_upa_chain_with_single_link() {
      mv::CompilationUnit &unit = upa_chain_with_single_link_;
      mv::OpModel& om = unit.model();
      
      auto input0 = om.input({1,1,1000,1}, mv::DType("Float64"),
          mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0#4");
      std::string axis = "C";
     
      auto softmax0 = om.softmax(input0, axis);
      auto softmax1 = om.softmax(softmax0, axis);
      auto softmax2 = om.softmax(softmax1, axis, mv::DType("Float64"),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}});

      std::vector<int64_t> weightsData0 =
          mv::utils::generateSequence<int64_t> (1*1*1000*1);
      auto weights0 = om.constantInt(weightsData0,{1,1,1000,1},
            mv::DType("UInt8"), mv::Order::getZMajorID(4),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}}, "conv#0_weights#1");

      auto conv0 = om.conv(softmax2, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1,
          mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}},
            "conv#10");

      auto softmax3 = om.softmax(conv0, axis);

      om.output(softmax3);

      std::string compDescPath(comp_desc_path_);
      unit.loadCompilationDescriptor(compDescPath);

      unit.loadTargetDescriptor(mv::Target::ma2490);
      unit.initialize();
      unit.run();
    }

    mv::OpModel& upa_chain_with_single_link() {
      return upa_chain_with_single_link_.model();
    }


    void setup_upa_chain_with_multiple_link() {
      mv::CompilationUnit &unit = upa_chain_with_multiple_link_;
      mv::OpModel& om = unit.model();

      auto input0 = om.input({1,1,500,1}, mv::DType("Float64"),
          mv::Order::getZMajorID(4), {{0},{1.0},{},{}}, "input:0#4");

      auto input1 = om.input({1,1,500,1}, mv::DType("Float64"),
          mv::Order::getZMajorID(4), {{0}, {1,0}, {}, {}}, "input:1#5");

      std::string axis = "C";

      auto implConcat = om.implicitConcat({input0,input1},axis, {{0},{1.0},{},{}}, "concat:2#1");

      auto softmax0 = om.softmax(implConcat, axis);
      auto softmax1 = om.softmax(softmax0, axis);
      auto softmax2 = om.softmax(softmax1, axis, mv::DType("Float64"),
            {{135},{0.0025439101736992598},{-0.3435550332069397},
              {0.3051420748233795}});
      auto softmax3 = om.softmax(softmax2, axis);

      om.output(softmax3);

      std::string compDescPath(comp_desc_path_);
      unit.loadCompilationDescriptor(compDescPath);

      unit.loadTargetDescriptor(mv::Target::ma2490);
      unit.initialize();
      unit.run();
    }
    
    mv::OpModel& upa_chain_with_multiple_link() {
      return upa_chain_with_multiple_link_.model();
    }

    mv::CompilationUnit three_layer_conv_input_;
    mv::CompilationUnit upa_chain_ending_with_dpu_;
    mv::CompilationUnit upa_chain_with_single_link_;
    mv::CompilationUnit upa_chain_with_multiple_link_;	
    std::string comp_desc_path_;
}; // class Operation_Dag_Test //


typedef mv::scheduler::Operation_Dag<mv::ControlModel> control_dag_t;
TEST_F(Operation_Dag_Test, test_structure_data_dag) {
  op_model_dag_t input(three_layer_conv_model());

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

  op = dag.get_op_by_name("Input_0");
  ASSERT_TRUE(op != NULL);
  itr = dag.begin_parent_nodes(op);
  itr_end = dag.end_parent_nodes(op);
  EXPECT_EQ(itr, itr_end);
}

TEST_F(Operation_Dag_Test, schedule_write_and_read_test) {
  typedef typename op_model_dag_t::operation_t operation_t;
  mv::OpModel &om = three_layer_conv_model();
  op_model_dag_t dag( om );
  size_t max_memory = 2000000;
  mv::scheduler::mv_lp_scheduler_t scheduler_begin(dag, max_memory),
      scheduler_end;

  typedef typename mv::lp_scheduler::Schedule_Reader_Writer<op_model_dag_t>
      writer_t;
  typedef typename mv::lp_scheduler::Schedule_Reader_Writer<op_model_dag_t>
      reader_t;

  std::ostringstream oss;
  bool status = writer_t::write_to_stringstream(oss, scheduler_begin,
      scheduler_end);

  // Read back the schedule //
  std::istringstream iss(oss.str());
  std::unordered_map<std::string, size_t> read_schedule;
  reader_t::schedule_read_iterator_t
      sbegin = reader_t::begin_read(iss, om), send;

  size_t record_count = 0UL;
  while (sbegin != send) {
    operation_t op = (operation_t) *sbegin;

    // Make sure the read op is in the OpModel //
    EXPECT_TRUE(om.getOp(op->getName()) != om.opEnd() );

    ++sbegin;
    ++record_count;
  }
  
  EXPECT_TRUE(status);
}

class Aligned_DMA_Test : public ::testing::Test {
  protected:

    void SetUp() override {
      std::string prefix(getenv("MCM_HOME"));
      comp_desc_path_ = prefix +
        "/config/compilation/release_kmb-sc.json";
      setup_aligned_dma_input();
    }

    void setup_aligned_dma_input() {
      using namespace mv;

      mv::CompilationUnit &unit = aligned_dma_input_;
      mv::OpModel& model = unit.model();
      //////////////////////////////////////////////////////////////////////////
    static const auto inf = std::numeric_limits<double>::infinity();
    const auto data_0 = model.input({56, 56, 96, 1}, mv::DType("UInt8"),
        mv::Order("NHWC"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}},
        "data");
    const auto ConstantInt_0_0 = model.constantInt(
        {
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1,1,1,1,1,
          1,1,1,1,1,1
        }, {96}, mv::DType("UInt8"), mv::Order("W"),
        {{0},{0.017429193481803, 0.017507003620267, 0.017057860270143},
        {-inf},{inf},{0, 0, 0},{1, 1, 1}}, "");
    const auto data_scaled = model.scale(data_0, ConstantInt_0_0,
        mv::DType("Default"), {{0},{1.00000000000000},{-inf},{inf},{0},{1}},
          "data_scaled");

    const std::vector<int64_t> ConstantInt_11_0_data(96, 1);
    const auto ConstantInt_11_0 = model.constantInt(ConstantInt_11_0_data,
        {96}, mv::DType("Int32"), mv::Order("W"), {{},{},{},{}}, "");
    const auto Add1_7055_Fused_Add__bias_0 = model.bias(data_scaled,
        ConstantInt_11_0, mv::DType("Default"), 
        {{1},{0.021110465750098},{-inf},{inf},{0},{1}},
        "Add1_7055/Fused_Add_:bias");
    const auto _328clamp_min_0 = model.minimum(Add1_7055_Fused_Add__bias_0,
        6.000000, mv::DType("Default"), 
        {{1},{0.021110465750098},{-inf},{inf},{0},{1}}, "328clamp-min");
    const auto _328clamp_max_0 = model.maximum(_328clamp_min_0, 0.000000,
        mv::DType("Default"), {{1},{0.021110465750098},{-inf},{inf},{0},{1}},
        "328clamp-max");

    const std::vector<int64_t> ConstantInt_12_0_data(2304,1);
    const auto ConstantInt_12_0 = model.constantInt(ConstantInt_12_0_data,
        {1, 1, 96, 24}, mv::DType("UInt8"), mv::Order("NCHW"),
        {{127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},{0.004956519231200, 0.004432321526110, 0.003990855999291, 0.005211046431214, 0.008438258431852, 0.008648737333715, 0.004849425982684, 0.003452872158960, 0.005038230679929, 0.003754003206268, 0.003950135782361, 0.007098698057234, 0.008037026971579, 0.007006390020251, 0.003703376511112, 0.005557639058679, 0.006851346231997, 0.005135843530297, 0.004725134000182, 0.004555922001600, 0.004107011947781, 0.004487436730415, 0.004674031399190, 0.004140492528677},{-inf},{inf},{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}, "");

    const auto Add1_7067_Fused_Add__0 = model.conv(_328clamp_max_0, ConstantInt_12_0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"), {{154},{0.069332063198090},{-inf},{inf},{0},{1}}, "Add1_7067/Fused_Add_");
    const std::vector<int64_t> ConstantInt_13_0_data(24, 1);
    const auto ConstantInt_13_0 = model.constantInt(ConstantInt_13_0_data, {24}, mv::DType("Int32"), mv::Order("W"), {{},{},{},{}}, "");
    const auto Add1_7067_Fused_Add__bias_0 = model.bias(Add1_7067_Fused_Add__0, ConstantInt_13_0, mv::DType("Default"), {{154},{0.069332063198090},{-inf},{inf},{0},{1}}, "Add1_7067/Fused_Add_:bias");
   
    const std::vector<int64_t> ConstantInt_14_0_data(3456,1);
    const auto ConstantInt_14_0 = model.constantInt(ConstantInt_14_0_data,
        {1, 1, 24, 144}, mv::DType("UInt8"), mv::Order("NCHW"), {{127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},{0.001408332725987, 0.001416480750777, 0.001374424784444, 0.001409875811078, 0.001550126471557, 0.002376043237746, 0.001981196226552, 0.002014759927988, 0.002245662966743, 0.001193709787913, 0.001337339635938, 0.002121845958754, 0.003068244550377, 0.000808176526334, 0.001908686361276, 0.000923887593672, 0.001182126579806, 0.002066894434392, 0.002756797475740, 0.001693514175713, 0.001212153350934, 0.003105913754553, 0.002527946839109, 0.001158370403573, 0.002914398442954, 0.001172742457129, 0.001276855240576, 0.000981418532319, 0.002252325182781, 0.002082723891363, 0.000657503551338, 0.001518967794254, 0.001250240486115, 0.001585448160768, 0.001260141958483, 0.001189971226268, 0.001446118811145, 0.000984183629043, 0.001390017918311, 0.002679276280105, 0.001257669646293, 0.001913205953315, 0.001372537459247, 0.000187428900972, 0.001734503894113, 0.002570166485384, 0.001162482658401, 0.002256092382595, 0.001284602447413, 0.001595217036083, 0.001754511496983, 0.001931336941198, 0.002307979157194, 0.001677585882135, 0.001643226249143, 0.001468747737817, 0.000928945373744, 0.001396510633640, 0.001415938721038, 0.001055946922861, 0.000943772902247, 0.000901064369828, 0.001192155410536, 0.001376010011882, 0.002434749621898, 0.001097133615986, 0.001418673316948, 0.000910610833671, 0.001427133218385, 0.001982713583857, 0.001488125766627, 0.001876306836493, 0.001845949911512, 0.001872191671282, 0.002865620190278, 0.000698571268003, 0.001807772903703, 0.001035894849338, 0.001512845978141, 0.001470141578466, 0.000836685823742, 0.000887622823939, 0.001248534070328, 0.001792407943867, 0.002102859783918, 0.002498092129827, 0.001098960638046, 0.002167284954339, 0.002292661461979, 0.001847162609920, 0.001604368444532, 0.000764060590882, 0.000369666173356, 0.001343069016002, 0.002672948874533, 0.000671466463245, 0.001317414105870, 0.001279351417907, 0.001682497793809, 0.001159871346317, 0.000516492582392, 0.001900690142065, 0.000567173410673, 0.001900458591990, 0.001160779385827, 0.001341296592727, 0.001591680338606, 0.001247186446562, 0.001743546687067, 0.003444514935836, 0.001697689760476, 0.002107800217345, 0.001846703002229, 0.000542576191947, 0.001702957903035, 0.000813523132820, 0.002212672028691, 0.002630191389471, 0.001300674048252, 0.001578363240696, 0.001708282390609, 0.001290857675485, 0.000961698126048, 0.002843917114660, 0.001830828841776, 0.002127258805558, 0.000949722656514, 0.001102125621401, 0.000414700334659, 0.002162177115679, 0.001297257142141, 0.002882179105654, 0.001283522578888, 0.002252145204693, 0.001271165674552, 0.001326248398982, 0.001024920260534, 0.001695366110653, 0.000427782593761, 0.002236638218164, 0.000888467649929, 0.001963876886293, 0.002358642406762, 0.002235516672954},{-inf},{inf},{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}, "");
    const auto Add1_7079_Fused_Add__0 = model.conv(Add1_7067_Fused_Add__bias_0, ConstantInt_14_0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"), {{0},{0.012730565853417},{-inf},{inf},{0},{1}}, "Add1_7079/Fused_Add_");

    const std::vector<int64_t> ConstantInt_15_0_data(144, 1);
    const auto ConstantInt_15_0 = model.constantInt(ConstantInt_15_0_data, {144}, mv::DType("Int32"), mv::Order("W"), {{},{},{},{}}, "");
    const auto Add1_7079_Fused_Add__bias_0 = model.bias(Add1_7079_Fused_Add__0, ConstantInt_15_0, mv::DType("Default"), {{0},{0.012730565853417},{-inf},{inf},{0},{1}}, "Add1_7079/Fused_Add_:bias");
    const auto _333clamp_min_0 = model.minimum(Add1_7079_Fused_Add__bias_0, 6.000000, mv::DType("Default"), {{0},{0.012730565853417},{-inf},{inf},{0},{1}}, "333clamp-min");
    const auto _333clamp_max_0 = model.maximum(_333clamp_min_0, 0.000000, mv::DType("Default"), {{0},{0.012730565853417},{-inf},{inf},{0},{1}}, "333clamp-max");

    const std::vector<int64_t> ConstantInt_16_0_data(1296, 1);
    const auto ConstantInt_16_0 = model.constantInt(ConstantInt_16_0_data,
        {3, 3, 144, 1}, mv::DType("UInt8"), mv::Order("NCHW"), {{127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},{0.013346560299397, 0.008176172152162, 0.012916129082441, 0.011116284877062, 0.007908622734249, 0.010800695978105, 0.015562847256660, 0.018119083717465, 0.015102543868124, 0.008644415065646, 0.009127425961196, 0.004701003432274, 0.007579009979963, 0.017160901799798, 0.010407056659460, 0.011404539458454, 0.011443682014942, 0.015116586349905, 0.011629979126155, 0.007943468168378, 0.022008281201124, 0.008727474138141, 0.006514991633594, 0.008995812386274, 0.011995472013950, 0.020743599161506, 0.014534511603415, 0.013651492074132, 0.015887882560492, 0.009767447598279, 0.017112666741014, 0.012015468440950, 0.011252827011049, 0.008115763776004, 0.007741436362267, 0.026618767529726, 0.014118491671979, 0.011179757304490, 0.006166183855385, 0.009445916861296, 0.010565783828497, 0.013344135135412, 0.005877207033336, 0.000431389024016, 0.012382653541863, 0.012761832214892, 0.015677437186241, 0.012986283749342, 0.009462713263929, 0.009613419882953, 0.009033069945872, 0.006162557750940, 0.010879754088819, 0.014668413437903, 0.021590195596218, 0.013621769845486, 0.007262078113854, 0.019800674170256, 0.016746098175645, 0.007740284316242, 0.015733070671558, 0.012996185570955, 0.014100674539804, 0.003456440754235, 0.012014987878501, 0.017479227855802, 0.014190479181707, 0.007675912696868, 0.007640562951565, 0.019977079704404, 0.010971932671964, 0.007772476878017, 0.011808281764388, 0.012286230921745, 0.004062911961228, 0.020050533115864, 0.009734938852489, 0.006064555142075, 0.011043137870729, 0.010719149373472, 0.015558176673949, 0.009974391199648, 0.008403891697526, 0.022948594763875, 0.009133218787611, 0.006010395474732, 0.008942591026425, 0.014516944065690, 0.010964754968882, 0.009871095418930, 0.006426933221519, 0.020747656002641, 0.014080727472901, 0.013502454385161, 0.013483911752701, 0.013208581134677, 0.020178202539682, 0.021575260907412, 0.012502184137702, 0.013164899311960, 0.008360309526324, 0.014877968467772, 0.023459294810891, 0.009668864309788, 0.021262343972921, 0.008495826274157, 0.006168255116791, 0.007920717820525, 0.016729265451431, 0.016553914174438, 0.014810425229371, 0.016385059803724, 0.006020342465490, 0.016261491924524, 0.009695622138679, 0.009803725406528, 0.003551279427484, 0.004305442795157, 0.007156666368246, 0.005123111885041, 0.013595796190202, 0.015162274241447, 0.027219092473388, 0.011691834777594, 0.019011968746781, 0.014373493380845, 0.022812036797404, 0.016630729660392, 0.043138902634382, 0.013745630159974, 0.016896499320865, 0.014589043334126, 0.014117149636149, 0.008725153282285, 0.008016566745937, 0.008279398083687, 0.016508279368281, 0.007296624127775, 0.025339189916849, 0.020323032513261, 0.011283368803561, 0.019368104636669, 0.003636420471594, 0.018112719058990},{-inf},{inf},{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}, "");
    const auto Add1_7091_Fused_Add__0 = model.depthwiseConv(_333clamp_max_0, ConstantInt_16_0, {1, 1}, {1, 1, 1, 1}, 1, mv::DType("Default"), {{1},{0.017471868544817},{-inf},{inf},{0},{1}}, "Add1_7091/Fused_Add_");


    const std::vector<int64_t> ConstantInt_17_0_data(144, 1);
    const auto ConstantInt_17_0 = model.constantInt(ConstantInt_17_0_data, {144}, mv::DType("Int32"), mv::Order("W"), {{},{},{},{}}, "");
    const auto Add1_7091_Fused_Add__bias_0 = model.bias(Add1_7091_Fused_Add__0, ConstantInt_17_0, mv::DType("Default"), {{1},{0.017471868544817},{-inf},{inf},{0},{1}}, "Add1_7091/Fused_Add_:bias");
    const auto _336clamp_min_0 = model.minimum(Add1_7091_Fused_Add__bias_0, 6.000000, mv::DType("Default"), {{1},{0.017471868544817},{-inf},{inf},{0},{1}}, "336clamp-min");
    const auto _336clamp_max_0 = model.maximum(_336clamp_min_0, 0.000000, mv::DType("Default"), {{1},{0.017471868544817},{-inf},{inf},{0},{1}}, "336clamp-max");

    const std::vector<int64_t> ConstantInt_18_0_data(3456, 1);
    const auto ConstantInt_18_0 = model.constantInt(ConstantInt_18_0_data, {1, 1, 144, 24}, mv::DType("UInt8"), mv::Order("NCHW"), {{127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},{0.009341502562165, 0.010417335666716, 0.007129909005016, 0.006068493239582, 0.005282321944833, 0.006712106056511, 0.005790446419269, 0.009849079884589, 0.005235156044364, 0.012592942453921, 0.008176170289516, 0.006280334200710, 0.005042237229645, 0.010481854900718, 0.006525698117912, 0.008093762211502, 0.004919426515698, 0.006496067624539, 0.010048791766167, 0.006126730237156, 0.005118255503476, 0.004911649972200, 0.006398214492947, 0.008445502258837},{-inf},{inf},{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}, "");
    const auto Add1_7103_Fused_Add__0 = model.conv(_336clamp_max_0, ConstantInt_18_0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"), {{101},{0.060786865651608},{-inf},{inf},{0},{1}}, "Add1_7103/Fused_Add_");
    const std::vector<int64_t> ConstantInt_19_0_data(24, 1);
    const auto ConstantInt_19_0 = model.constantInt(ConstantInt_19_0_data, {24}, mv::DType("Int32"), mv::Order("W"), {{},{},{},{}}, "");
    const auto Add1_7103_Fused_Add__bias_0 = model.bias(Add1_7103_Fused_Add__0, ConstantInt_19_0, mv::DType("Default"), {{101},{0.060786865651608},{-inf},{inf},{0},{1}}, "Add1_7103/Fused_Add_:bias");
    const auto _339_0 = model.eltwise({Add1_7067_Fused_Add__bias_0, Add1_7103_Fused_Add__bias_0}, "Add", mv::DType("Default"), {{120},{0.065235227346420},{-inf},{inf},{0},{1}}, "339");

    const std::vector<int64_t> ConstantInt_20_0_data(3456, 1);
    const auto ConstantInt_20_0 = model.constantInt(ConstantInt_20_0_data,
        {1, 1, 24, 144}, mv::DType("UInt8"), mv::Order("NCHW"), {{127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},{0.001289694919251, 0.001134309684858, 0.001015982124954, 0.000754617736675, 0.001588942250237, 0.000500845897477, 0.001299780327827, 0.000399871671107, 0.000890520983376, 0.001384254079312, 0.000937096949201, 0.000901810824871, 0.000988836283796, 0.000718487601262, 0.001145184971392, 0.000530713470653, 0.000381910533179, 0.001214338000864, 0.000991224544123, 0.001040934934281, 0.001090044854209, 0.001333619817160, 0.001017095986754, 0.001150571159087, 0.001402063644491, 0.000656555057503, 0.001835674745962, 0.002112473128363, 0.000977189862169, 0.002183666918427, 0.001309497980401, 0.000933982315473, 0.000896110315807, 0.001285708392970, 0.001046616467647, 0.002168192993850, 0.001056316075847, 0.001494414638728, 0.001439549378119, 0.001526652136818, 0.001215883181430, 0.001177303842269, 0.000489173980895, 0.001716623315588, 0.000847760762554, 0.001074683852494, 0.000870996038429, 0.001655776053667, 0.001327547011897, 0.000988040235825, 0.001287658815272, 0.001779338344932, 0.001427155919373, 0.000984766753390, 0.001633922685869, 0.002265664050356, 0.001085851690732, 0.001342286588624, 0.001576389535330, 0.000971319444943, 0.001679972396232, 0.000889574934263, 0.000927822024096, 0.000580395804718, 0.000554160564207, 0.001004101242870, 0.001472359057516, 0.001341656548902, 0.001266256440431, 0.000427994207712, 0.001092127640732, 0.000444349250756, 0.001911417231895, 0.001030594925396, 0.001361211063340, 0.001067007542588, 0.001090719713829, 0.000890973024070, 0.001451190444641, 0.000750800012611, 0.000958382675890, 0.001935779233463, 0.002309277886525, 0.001140703330748, 0.002704317914322, 0.001206665416248, 0.000971885863692, 0.000637648859993, 0.000993532361463, 0.001149474410340, 0.000740003073588, 0.000783607014455, 0.001205612556078, 0.001692878664471, 0.001329992315732, 0.001857607625425, 0.001197124249302, 0.000538095890079, 0.000868107017595, 0.000498513458297, 0.001399367465638, 0.001438391162083, 0.001266842475161, 0.000912111834623, 0.001052813953720, 0.001148147624917, 0.000473302992759, 0.001554419985041, 0.001633433043025, 0.001439448446035, 0.001122021698393, 0.001194763113745, 0.000578333332669, 0.001719470950775, 0.001142575056292, 0.001344473101199, 0.002204052871093, 0.000949189474341, 0.001181079074740, 0.001865879399702, 0.001041064155288, 0.001131558790803, 0.000792417908087, 0.001383253256790, 0.001035781693645, 0.000721288030036, 0.001890393788926, 0.000842604553327, 0.000304436631268, 0.000614403106738, 0.000917831668630, 0.000516136758961, 0.000296142854495, 0.001616345020011, 0.001404309645295, 0.000890694209374, 0.000909045105800, 0.001600829651579, 0.000804976967629, 0.001316471607424, 0.002359694335610, 0.001330786151811, 0.001398414373398, 0.001034506829455},{-inf},{inf},{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}, "");
    const auto Add1_7115_Fused_Add__0 = model.conv(_339_0, ConstantInt_20_0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"), {{0},{0.018329823389649},{-inf},{inf},{0},{1}}, "Add1_7115/Fused_Add_");
    const auto output = model.output(Add1_7115_Fused_Add__0, mv::DType("Default"), {{},{},{},{}}, true, "");
      /////////////////////////////////////////////////////////////////////////

      std::string compDescPath(comp_desc_path_);
      unit.loadCompilationDescriptor(compDescPath);

      unit.loadTargetDescriptor(mv::Target::ma2490);
      unit.initialize();
      unit.run();
    }

    mv::OpModel& get_aligned_dma_model() { 
      return aligned_dma_input_.model();
    }

    Aligned_DMA_Test() : aligned_dma_input_("aligned_dma_input_model"), 
      comp_desc_path_() {}

    mv::CompilationUnit aligned_dma_input_;
    std::string comp_desc_path_;
}; // class Aligned_DMA_Test //


TEST_F(Aligned_DMA_Test, is_aligned_dma_op) {

  mv::OpModel& om = get_aligned_dma_model();
  op_model_dag_t input_dag(om);

  // Test all branches in the method is_aligned_dma_op //

  // Typical case : Aligned DMA//
  EXPECT_TRUE(input_dag.is_aligned_dma_op(om, "339:0_align_copy0"));
  // DMA but not aligned //
  EXPECT_FALSE(input_dag.is_aligned_dma_op(om,
          "ConstantInt_10_ALIGNED_DDR2CMX"));
  //non DMA op //
  EXPECT_FALSE(input_dag.is_aligned_dma_op(om, "Add1_7091/Fused_Add_"));

  //DMA but write //
  EXPECT_FALSE(input_dag.is_aligned_dma_op(om,
          "Add1_7067/Fused_Add__NNCMX2DDR"));
}

TEST_F(Aligned_DMA_Test, resource_utility) {
  mv::OpModel& om = get_aligned_dma_model();
  op_model_dag_t input_dag(om);


  // Typical case : Aligned DMA//
  EXPECT_EQ(input_dag.resource_utility(om, "339:0_align_copy0"), 100352UL);

  // DMA but not aligned //
  EXPECT_EQ(input_dag.resource_utility(om, "ConstantInt_10_ALIGNED_DDR2CMX"),
      4608UL);

  EXPECT_EQ(input_dag.resource_utility(om, "Add1_7091/Fused_Add_"), 451584UL);
}

namespace mv_unit_testing {

class Barrier_Control_Dag_Test : public Operation_Dag_Test {

  protected:
    ////////////////////////////////////////////////////////////////////////////
    typedef mv::lp_scheduler::Control_Model_Barrier_Scheduler
        barrier_scheduler_t;
    typedef typename barrier_scheduler_t::barrier_control_edge_t
      barrier_control_edge_t;
    typedef typename barrier_scheduler_t::operation_t operation_t;
    typedef typename std::list<operation_t> op_chain_t;
    ////////////////////////////////////////////////////////////////////////////


    void SetUp() override {
      std::string prefix(getenv("MCM_HOME"));
      comp_desc_path_ = prefix + "/config/compilation/release_kmb-sc.json";
      setup_three_layer_conv_input();
    }

    op_chain_t get_tailing_upa_chain(
        const barrier_scheduler_t& barrier_scheduler) const {
      op_chain_t chain;
      barrier_scheduler.get_tailing_upa_chain(std::back_inserter(chain));
      return chain;
    }
}; // class Barrier_Control_Dag_Test //

} // namespace mv_unit_testing

using namespace mv_unit_testing;

TEST_F(Barrier_Control_Dag_Test, barrier_control_dag) {
  mv::OpModel& om = three_layer_conv_model();
  mv::ControlModel cm(om); 

  clear_control_model_and_remove_barriers(om);

  // run the barrier scheduler on the underlying opmodel//
  barrier_scheduler_t barrier_scheduler(cm, 4UL, 256UL);

  barrier_scheduler.schedule();

  // the setup already runs the barrier scheduler //
  EXPECT_TRUE(is_valid_barrier_schedule(cm));
  EXPECT_TRUE(does_barrier_ids_have_no_gaps(cm));
}

TEST_F(Barrier_Control_Dag_Test, upa_chain_ending_with_dpu) {
  setup_upa_chain_ending_with_dpu_model();

  mv::OpModel &om = upa_chain_ending_with_dpu_model(); 
  mv::ControlModel cm(om); 

  clear_control_model_and_remove_barriers(om);

  // run the barrier scheduler on the underlying opmodel//
  barrier_scheduler_t barrier_scheduler(cm, 4UL, 256UL);

  barrier_scheduler.schedule();

  op_chain_t upa_chain = get_tailing_upa_chain(barrier_scheduler);

  EXPECT_TRUE(upa_chain.empty());

  EXPECT_TRUE(is_valid_barrier_schedule(cm));
  EXPECT_TRUE(does_barrier_ids_have_no_gaps(cm));
}

TEST_F(Barrier_Control_Dag_Test, upa_chain_with_single_link) {
  setup_upa_chain_with_single_link();

  mv::OpModel &om = upa_chain_with_single_link(); 
  mv::ControlModel cm(om); 

  clear_control_model_and_remove_barriers(om);

  // run the barrier scheduler on the underlying opmodel//
  barrier_scheduler_t barrier_scheduler(cm, 4UL, 256UL);

  barrier_scheduler.schedule();

  op_chain_t upa_chain = get_tailing_upa_chain(barrier_scheduler);

  // compiler puts a quantize before softmax and two barriers//
  ASSERT_EQ(upa_chain.size(), 4UL);

  barrier_scheduler.remove_barriers_in_upa_chain_connected_to_output();
  EXPECT_TRUE(does_barrier_ids_have_no_gaps(cm));
}

TEST_F(Barrier_Control_Dag_Test, upa_chain_with_multiple_link) {
  setup_upa_chain_with_multiple_link();

  mv::OpModel &om = upa_chain_with_multiple_link();
  mv::ControlModel cm(om);

  clear_control_model_and_remove_barriers(om);

  // run the barrier scheduler on the underlying opmodel//
  barrier_scheduler_t barrier_scheduler(cm, 7UL, 256UL);

  barrier_scheduler.schedule();

  op_chain_t upa_chain = get_tailing_upa_chain(barrier_scheduler);

  // compiler puts a quantize before softmax and two barriers//
  ASSERT_EQ(upa_chain.size(), 7UL);

  barrier_scheduler.remove_barriers_in_upa_chain_connected_to_output();
  EXPECT_TRUE(does_barrier_ids_have_no_gaps(cm));
}
