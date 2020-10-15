#include "gtest/gtest.h"

#include "include/mcm/compiler/compilation_unit.hpp"
#include "pass/lp_scheduler/cmx_concat_transform.hpp"
#include "pass/lp_scheduler/operation_precedence_dag.hpp"
#include "pass/lp_scheduler/lp_scheduler_pass.hpp"

class CMX_Concatenation_Test : public ::testing::Test {
  public:
    typedef mv::scheduler::CMX_Concatenation cmx_concatenation_t;
    typedef typename cmx_concatenation_t::concat_subgraph_t concat_subgraph_t;
    typedef typename cmx_concatenation_t::control_edge_t control_edge_t;
    typedef mv::scheduler::Operation_Dag<mv::OpModel> op_model_dag_t;
    typedef mv::lp_scheduler::mv_memory_scheduler_with_spilling_t scheduler_t;
    typedef typename scheduler_t::scheduled_op_info_t scheduled_op_info_t;


    CMX_Concatenation_Test()
      : cmx_concatable_("positive"), cmx_non_concatable_("negative") {}

  protected:


    void SetUp() override {}
    void TearDown() override {}

    void remove_all_barrier_tasks(mv::OpModel& om) {
      std::vector<mv::Data::OpListIterator> barrier_ops;
      for (auto oitr=om.opBegin(); oitr!=om.opEnd(); ++oitr) {
        if (oitr->getOpType() == "BarrierTask") {
          barrier_ops.push_back(oitr);
        }
      }

      for (auto ooitr=barrier_ops.begin(); ooitr!=barrier_ops.end(); ++ooitr) {
        om.removeOp(*ooitr);
      }
    }

    void set_default_quant_params_for_all_tensors(mv::OpModel& om) {
      mv::QuantizationParams qparams({{0}, {0}, {0.0}, {1.0}});
      for (auto oitr=om.opBegin(); oitr!=om.opEnd(); ++oitr) {
        printf("name=%s\n", oitr->getName().c_str());
        oitr->set<mv::QuantizationParams>("quantParams", qparams);
      }
    }

    mv::OpModel& create_cmx_concatable_test() {
      std::string prefix(getenv("MCM_HOME"));
      std::string comp_desc_path = prefix +
          "/config/compilation/release_kmb-sc.json";
      mv::CompilationUnit &unit = cmx_concatable_;
      mv::OpModel &om = unit.model();
      double inf = std::numeric_limits<double>::infinity();

      ////////////////////////////[MODEL BEGIN] ////////////////////////////////
      auto input0 = om.input({227,227,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{1.0},{-128.0},{127.0}}, "true, input#89");

      std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*64);
      auto weights0 = om.constantInt(weightsData0,{3,3,3,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.006698549259454012},{-0.8641261458396912},{0.8373053669929504}}, "conv1/Relu#0_weights#1");
      auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{3.6097023487091064},{0.0},{920.47412109375}}, "conv1/Relu#90");

      std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (64);
      auto biasWeights0 = om.constantInt(biasWeightsData0,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.006698549259454012},{-inf},{inf}}, "conv1/Relu#0_bias#2");
      auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{3.6097023487091064},{0.0},{920.47412109375}});


      auto pool0 = om.maxPool(bias_c0, {3, 3}, {2, 2}, {0, 0, 0, 0}, false, mv::DType("UInt8"), {{0},{3.6097023487091064},{0.0},{920.47412109375}}, "pool1/MaxPool#91");

      std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (1*1*64*16);
      auto weights1 = om.constantInt(weightsData1,{1,1,64,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.0099817318841815},{-1.337534785270691},{1.1978250741958618}}, "fire2/squeeze1x1/Relu#4_weights#5");
      auto conv1 = om.conv(pool0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{7.038242816925049},{0.0},{1794.751953125}}, "fire2/squeeze1x1/Relu#92");

      std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (16);
      auto biasWeights1 = om.constantInt(biasWeightsData1,{16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0360310822725296},{-inf},{inf}}, "fire2/squeeze1x1/Relu#4_bias#6");
      auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"), {{0},{7.038242816925049},{0.0},{1794.751953125}});

      std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (1*1*16*64);
      auto weights2 = om.constantInt(weightsData2,{1,1,16,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.006223201286047697},{-0.709420919418335},{0.8712722659111023}}, "fire2/expand1x1/Relu#7_weights#8");
      auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{5.427421569824219},{0.0},{1383.9925537109375}}, "fire2/expand1x1/Relu#93");

      std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
      auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.04380040243268013},{-inf},{inf}}, "fire2/expand1x1/Relu#7_bias#9");
      auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"), {{0},{5.427421569824219},{0.0},{1383.9925537109375}});

      std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (3*3*16*64);
      auto weights3 = om.constantInt(weightsData3,{3,3,16,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.005374675616621971},{-0.7201219797134399},{0.6450456380844116}}, "fire2/expand3x3/Relu#10_weights#11");
      auto conv3 = om.conv(bias_c1, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{5.427421569824219},{0.0},{1383.9925537109375}}, "fire2/expand3x3/Relu#94");

      std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (64);
      auto biasWeights3 = om.constantInt(biasWeightsData3,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.03782827407121658},{-inf},{inf}}, "fire2/expand3x3/Relu#10_bias#12");
      auto bias_c3 = om.bias(conv3, biasWeights3, mv::DType("UInt8"), {{0},{5.427421569824219},{0.0},{1383.9925537109375}});

      auto concat0 = om.concat({bias_c2, bias_c3}, "C", mv::DType("UInt8"), {{0},{5.427421569824219},{0.0},{1383.9925537109375}}, "concat#95");

      std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (1*1*128*16);
      auto weights4 = om.constantInt(weightsData4,{1,1,128,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{136},{0.00555797666311264},{-0.7503653168678284},{0.6613607406616211}}, "fire3/squeeze1x1/Relu#14_weights#15");
      auto conv4 = om.conv(concat0, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{6.281661510467529},{0.0},{1601.82373046875}}, "fire3/squeeze1x1/Relu#96");
      om.output(conv4);
      ////////////////////////////[MODEL END] //////////////////////////////////

      mv::GenerateDotFromModel(om, "OpModel", "input_opmodel_orig.dot");

      unit.loadCompilationDescriptor(comp_desc_path);
      unit.loadTargetDescriptor(mv::Target::ma2490);
      unit.initialize();
      unit.run();
      remove_all_barrier_tasks(unit.model());
      return unit.model();
    }


    mv::CompilationUnit cmx_concatable_;
    mv::CompilationUnit cmx_non_concatable_;
}; // class CMX_Concatenation_Test //


TEST_F(CMX_Concatenation_Test, basic_test) {
  mv::OpModel& om = create_cmx_concatable_test();
  mv::GenerateDotFromModel(om, "OpModel", "input_opmodel.dot");

  cmx_concatenation_t cmx_concat_algo(om);

  std::list<concat_subgraph_t> concat_subgraphs;

  cmx_concat_algo.locate_concat_subgraphs(std::back_inserter(concat_subgraphs));
  ASSERT_FALSE(concat_subgraphs.empty());
  // model has 3 concats //
  EXPECT_EQ(concat_subgraphs.size(), 3UL);

  size_t cmxable_concats = 0UL;
  for (auto gitr=concat_subgraphs.begin(); gitr!=concat_subgraphs.end(); ++gitr)
  {
     if (gitr->is_cmx_concateable()) { cmxable_concats++; }
     gitr->dump();
  }
  // 2 of them are CMX concateable //
  EXPECT_EQ(cmxable_concats, 2UL);
}

TEST_F(CMX_Concatenation_Test, control_edges) {
  mv::OpModel& om = create_cmx_concatable_test();

  cmx_concatenation_t cmx_concat_algo(om);

  std::list<concat_subgraph_t> concat_subgraphs;

  cmx_concat_algo.locate_concat_subgraphs(std::back_inserter(concat_subgraphs));
  ASSERT_FALSE(concat_subgraphs.empty());
  // model has 3 concats //
  EXPECT_EQ(concat_subgraphs.size(), 3UL);

  size_t cmxable_concats = 0UL;
  for (auto gitr=concat_subgraphs.begin(); gitr!=concat_subgraphs.end(); ++gitr)
  {
     concat_subgraph_t& concat_subgraph = *gitr;
     if (gitr->is_cmx_concateable()) {
       cmxable_concats++;
       // total concat edges = dpu_in_.size() - 1UL + dpu_out_.size() //
       std::list<control_edge_t> control_edges;
       size_t total_count =
          cmx_concat_algo.transform_and_get_control_edges(concat_subgraph,
                std::back_inserter(control_edges));
       ASSERT_EQ( total_count,
            (gitr->dpu_in_.size() + gitr->dpu_out_.size() - 1UL));
       ASSERT_EQ( total_count, control_edges.size());
       for (auto citr=control_edges.begin(); citr!=control_edges.end(); ++citr){
         ASSERT_EQ(citr->source_itr_->getName(),
             (gitr->representative_dpu_)->getName()); 
       }
     }
  }
  // 2 of them are CMX concateable //
  EXPECT_EQ(cmxable_concats, 2UL);
}

TEST_F(CMX_Concatenation_Test, meta_data_transfer_to_opmodel) {
  mv::OpModel& om = create_cmx_concatable_test();
  cmx_concatenation_t cmx_concat_algo(om);

  std::list<control_edge_t> control_edges;
  cmx_concat_algo.transform_op_model(std::back_inserter(control_edges));

  EXPECT_EQ(control_edges.size(), 4UL);
}

TEST_F(CMX_Concatenation_Test, run_scheduler) {
  mv::OpModel& om = create_cmx_concatable_test();
  mv::GenerateDotFromModel(om, "OpModel", "input_opmodel_orig_sched.dot");
  cmx_concatenation_t cmx_concat_algo(om);

  std::list<control_edge_t> control_edges;
  cmx_concat_algo.transform_op_model(std::back_inserter(control_edges));

  // STEP-1: create op dag //
  op_model_dag_t dag(om);

  // STEP-2: update Operation_Dag //
  dag.enable_cmx_concat_transforms(om, 917504UL);

  // STEP-3: run scheduler //
  size_t max_memory = 917504UL;
  scheduler_t scheduler_begin(dag, max_memory), scheduler_end;

  std::unordered_map<size_t, size_t> memory_use_map;
  size_t make_span = 0UL;
  while (scheduler_begin != scheduler_end) {
    scheduled_op_info_t scheduled_op = *scheduler_begin;
    printf("op = %-20s  type = %-15s  time = %lu",
        (scheduled_op.op_)->getName().c_str(), scheduled_op.op_type_name(),
        scheduled_op.time_);
    if (scheduled_op.has_active_resource()) {
      printf( " resource=[%lu %lu]\n", scheduled_op.begin_resource(),
          scheduled_op.end_resource());
    } else {
      printf( " resource=<none>\n");
    }
    ++scheduler_begin;
    make_span = std::max(make_span, scheduled_op.time_);
  }

  // Make span reduces from 25 to 23 as reads and writes are removed from the
  // concat //
  EXPECT_EQ(make_span, 23UL);
}
