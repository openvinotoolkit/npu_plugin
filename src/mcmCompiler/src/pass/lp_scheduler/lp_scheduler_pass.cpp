#include <time.h>
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "lp_scheduler/lp_scheduler_pass.hpp"
#include "pass/lp_scheduler/control_edge_generator.hpp"
#include "scheduler/feasible_scheduler.hpp"


static void LpSchedulerPass(const mv::pass::PassEntry& , mv::ComputationModel&,
    mv::TargetDescriptor& , mv::Element&, mv::Element&);

namespace mv {
namespace pass {

MV_REGISTER_PASS(LpScheduler)
  .setFunc(LpSchedulerPass)
  .defineArg(json::JSONType::String, "output")
  .setDescription("Run Feasible Scheduler Algorithm.");

} // namespace mv //
} // namespace pass //

typedef mv::scheduler::Operation_Dag<mv::ControlModel> control_dag_t;
typedef mv::scheduler::Operation_Dag<mv::OpModel> dag_t;
typedef typename dag_t::operation_t operation_t;
typedef mv::lp_scheduler::Scheduled_Op scheduled_op_t;
typedef mv::lp_scheduler::Control_Edge control_edge_t;
typedef mv::lp_scheduler::Control_Edge_Set control_edge_set_t;
typedef mv::lp_scheduler::Control_Edge_Generator<scheduled_op_t>
  control_edge_generator_t;

class lp_scheduler_exception_t : std::string {
  public:
    lp_scheduler_exception_t(const std::string& msg) : std::string(msg) {}
    lp_scheduler_exception_t(const char *msg) : std::string(msg) {}
    const std::string& getMessage() const { return  *this; }
}; // class lp_scheduler_exception_t //


void LpSchedulerAllocatorPass(mv::ComputationModel& model,
      mv::Element& passDesc) {
  typedef typename mv::lp_scheduler::Schedule_Reader_Writer<dag_t> reader_t;
  mv::lp_scheduler::Tensor_Allocator_Assignment alloc(model);
  mv::OpModel om(model);
  auto global_params = model.getGlobalConfigParams();

  if (global_params->hasAttr(reader_t::ddr_address_attribute())) {
    dag_t input_dag(om);
    typename reader_t::schedule_read_iterator_t begin, end;
    bool add_ddr_control_edges = (passDesc.hasAttr("ddr_control_edges") &&
        passDesc.get<bool>("ddr_control_edges"));

    mv::lp_scheduler::Master_Slave_Buffer_Relations<dag_t>
        msrelations(input_dag, model);

    const std::string& stringdata =
      global_params->get<std::string>(reader_t::ddr_address_attribute());
    assert(!stringdata.empty());

    const char *lp_sched_ddr_address_dump_filename = nullptr;
    if (mv::isDebugFilesEnabled())
    {
      lp_sched_ddr_address_dump_filename = "lp_sched_ddr_address_dump.txt";
    }
    std::istringstream iss(stringdata);
    begin = reader_t::begin_read(iss, om);
    end = reader_t::end_read();

    mv::lp_scheduler::DDR_Address_Generator<dag_t> ddr_address_generator(
        model, input_dag, std::numeric_limits<int>::max());
    bool status = ddr_address_generator.generate_tensor_addresses(begin, end,
        lp_sched_ddr_address_dump_filename, add_ddr_control_edges);
    if (!status) {
      throw std::string("[DDR_Address_Generation]: insufficient DDR space");
    }
  }

  for (auto itr=om.opBegin(); itr!=om.opEnd(); ++itr) {
    mv::Op &op = *itr;
    if (!op.outputSlots()) { continue; }

    mv::Data::TensorIterator tensor_itr = op.getOutputTensor(0UL);
    alloc(tensor_itr);
  }

}


void LpSchedulerBuildTimeStamp(FILE *fptr) {
  assert(fptr);
  fprintf(fptr, "[LpScheduler: build %s %s]\n", __DATE__, __TIME__);
}

void LpSchedulerPass(const mv::pass::PassEntry& pass,
    mv::ComputationModel& model, mv::TargetDescriptor&,
    mv::Element& passDesc, mv::Element&) {
  typedef mv::lp_scheduler::mv_memory_scheduler_with_spilling_t scheduler_t;
  typedef typename scheduler_t::scheduled_op_info_t scheduled_op_info_t;
  typedef mv::lp_scheduler::scheduler_traits<dag_t> traits_t;

  if (passDesc.hasAttr("allocator_mode")) {
    LpSchedulerAllocatorPass(model, passDesc);
    return;
  }


  mv::OpModel cm(model);
  dag_t input_dag;
  auto params = model.getGlobalConfigParams();

  dag_t::resource_t upper_bound = params->get<unsigned>("totalCmx");
  printfInfo("LpScheduler:", "[upper_bound = %lu]\n", upper_bound);

  // update the operation precedence dag with CMX concat transforms //
  bool apply_cmx_concat_transforms = passDesc.hasAttr("enable_cmx_concat") &&
    passDesc.get<bool>("enable_cmx_concat");
  typedef typename mv::scheduler::CMX_Concatenation::control_edge_t
      cmx_concat_control_edge_t;

  std::list<cmx_concat_control_edge_t> cmx_concat_control_edges;

  if (apply_cmx_concat_transforms) {
    //mv::GenerateDotFromModel(cm, "OpModel", "opmodel_before_transform.dot");

    std::string ignore_these_concats;
    if (passDesc.hasAttr("ignore_these_concats")) {
      ignore_these_concats = passDesc.get<std::string>("ignore_these_concats");
    }
    input_dag.enable_cmx_concat_transforms(cm, cmx_concat_control_edges,
          upper_bound, ignore_these_concats);
    //mv::GenerateDotFromModel(cm, "OpModel", "opmodel_after_transform.dot");
  } else {
    input_dag.reset(cm);
  }

  if (passDesc.hasAttr("enable_inplace_eltwise")) {
    input_dag.enable_eltwise_transforms(cm);
  }

  FILE *fptr = nullptr;
  if (mv::isDebugFilesEnabled()) {
    const std::string output_file = passDesc.get<std::string>("output");
    fptr = fopen(output_file.c_str(), "w");
    assert(fptr);
  }

  // precondition check //
  {
    std::list< std::pair<dag_t::operation_t, size_t> > exceeding_ops;

    input_dag.find_all_ops_exceeding_resource_threshold(upper_bound,
        std::back_inserter(exceeding_ops));

    for (auto itr=exceeding_ops.begin(); itr!=exceeding_ops.end(); ++itr) {
      pass.log(mv::Logger::MessageType::Info,
          " Exceeding Op: " + (itr->first)->getName() +
          " with resources:# " + (std::to_string(itr->second)));
    }

    if (!exceeding_ops.empty()) {
      fprintf(stderr, "exceeding ops %lu\n", exceeding_ops.size());
      fflush(stderr);
      if (fptr)
        fclose(fptr);
      throw "Exceeding ops ";
    }
  }

  if (fptr) {
    LpSchedulerBuildTimeStamp(fptr);
  }

  // generate tensor addresses //
  mv::lp_scheduler::Tensor_Address_Assignment<scheduler_t>
      cmx_address_alloc(model);
  scheduler_t scheduler(input_dag, upper_bound), scheduler_end;
  typedef std::list<scheduled_op_t> scheduled_op_list_t;
  typedef typename scheduled_op_list_t::iterator scheduled_op_list_iterator_t;
  scheduled_op_list_t scheduled_ops;

  clock_t scheduler_algo_start_time = clock();
  std::string scheduled_op_type; 
  while (scheduler != scheduler_end) { // collect original schedule //
    const scheduled_op_info_t &scheduled_op = *scheduler;
    mv::Op const *op = scheduled_op.op_;
    size_t rbegin = scheduled_op.begin_resource();
    size_t rend = scheduled_op.end_resource();

    scheduled_op_type = scheduled_op.op_type_name();

    if (input_dag.is_input_op(op)) {
      // explicitly set the resource bounds so that the prefetch edges can 
      // be done as high as possible.
      rbegin = 1UL;
      rend = upper_bound;
    }

    mv::lp_scheduler::op_type_e scheduled_op_enum;
    if (op->getOpType() != "PseudoOp") {
      if (scheduled_op_type == "ORIGINAL") {
        scheduled_op_enum = mv::lp_scheduler::op_type_e::ORIGINAL_OP;
      } else if (scheduled_op_type == "SPILLED_READ") {
        scheduled_op_enum = mv::lp_scheduler::op_type_e::SPILLED_READ_OP;
      } else {
        scheduled_op_enum = mv::lp_scheduler::op_type_e::SPILLED_WRITE_OP;
      }
      scheduled_ops.push_back(scheduled_op_t(op, scheduled_op.time_,
            rbegin, rend, scheduled_op_enum));
    }
    ++scheduler;
  } // while (scheduler != scheduler_end) //

  { // remove any pseudo ops //
    std::list<mv::Data::OpListIterator> ops_to_remove;
    for (mv::Data::OpListIterator oitr=cm.opBegin(); oitr!=cm.opEnd();
          ++oitr) {
      if (oitr->getOpType() == "PseudoOp") {
        ops_to_remove.push_back(oitr);
      }
    }

    for (mv::Data::OpListIterator oitr : ops_to_remove) {
      input_dag.remove_op_from_dag(&(*oitr));
      cm.removeOp(oitr);
    }
  }

  // drop all pseudo edges from input_dag and model //
  {
    input_dag.drop_all_pseudo_edges_in_model(cm);
    input_dag.drop_all_pseudo_edges();
  }


  { 
    std::list<scheduled_op_t> new_scheduled_ops;

    mv::lp_scheduler::Remove_Redundant_Spill_Writes::remove(
        scheduled_ops.begin(), scheduled_ops.end(),
          std::back_inserter(new_scheduled_ops));
    scheduled_ops = new_scheduled_ops;
  }


  bool has_any_dynamic_spill_ops = false;
  {
    for (auto shed_itr=scheduled_ops.begin(); shed_itr!=scheduled_ops.end();
        ++shed_itr) {
      const scheduled_op_t& scheduled_op = *shed_itr;

      cmx_address_alloc(scheduled_op);

      if (!scheduled_op.is_original_op()) { has_any_dynamic_spill_ops = true; }

      //////////////////////////////////debug///////////////////////////////////
      if (fptr) {
        fprintf(fptr, "op = %-20s  type = %-15s  time = %lu",
            (scheduled_op.op_)->getName().c_str(), scheduled_op.op_type_name(),
              scheduled_op.schedule_time_);
        fflush(fptr);

        if (scheduled_op.has_active_resource()) {
          fprintf(fptr, " resource=[%lu %lu] size=%lu\n",
              scheduled_op.cmx_address_start_, scheduled_op.cmx_address_end_,
         (scheduled_op.cmx_address_end_ - scheduled_op.cmx_address_start_)+1UL);
        } else {
          fprintf(fptr, " resource=<none>\n");
        }
      }
      //////////////////////////////////////////////////////////////////////////
    }
  }


  std::vector<control_edge_t> dynamic_spill_control_edges;
  if (has_any_dynamic_spill_ops) {
    printfInfo("LpScheduler:",
        "[Dynamic_Spill_Node_Inserter] adding dynamic spill nodes\n");
    mv::lp_scheduler::Dynamic_Spill_Node_Inserter<dag_t> dynamic_spill(
          input_dag, model);

    dynamic_spill.add_spill_read_write_ops(scheduled_ops.begin(),
          scheduled_ops.end());

    // Update the scheduled ops with new info of the ops //
    dynamic_spill.update_scheduled_ops_with_new_read_write_ops(
        scheduled_ops.begin(), scheduled_ops.end(),
        std::back_inserter(dynamic_spill_control_edges));

    dynamic_spill.generate_control_edges_for_spilled_cmx_concat_ops(
        std::back_inserter(dynamic_spill_control_edges));

    { // Erase any redundant spilled writes //
      scheduled_op_list_iterator_t sched_itr=scheduled_ops.begin(),
                                   sched_itr_next;
      while (sched_itr != scheduled_ops.end()) {
        sched_itr_next = sched_itr; ++sched_itr_next;
        if ((*sched_itr).op_ == NULL) {
          scheduled_ops.erase(sched_itr);
        }
        sched_itr = sched_itr_next;
      }
    }
    if (fptr) {
      dynamic_spill.print_redundant_spilled_ops(fptr);
    }

    // update the input_dag with updated opModel
    mv::OpModel updated_om(model);
    if (apply_cmx_concat_transforms) {
      //mv::GenerateDotFromModel(updated_om, "OpModel",
       //     "opmodel_before_spilled_transform.dot");
      input_dag.reset_from_cmx_concat_control_edges(updated_om,
            cmx_concat_control_edges);
      //mv::GenerateDotFromModel(updated_om, "OpModel",
       //     "opmodel_after_spilled_transform.dot");
    } else {
      input_dag.reset(updated_om);
    }

    // updated schedule //
    if (fptr) {
      fprintf(fptr, "\n\n\n======================\n");
    }
    for (auto sitr=scheduled_ops.begin(); sitr!=scheduled_ops.end(); ++sitr) {
      const scheduled_op_t& scheduled_op = *sitr;
      assert( scheduled_op.is_original_op() );
      if (fptr) {
        fprintf(fptr, "[updated] op = %-20s  type = %-15s  time = %lu ",
            (scheduled_op.op_)->getName().c_str(), scheduled_op.op_type_name(),
            scheduled_op.schedule_time_);

        if (scheduled_op.has_valid_address()) {
          fprintf(fptr, " resource=[%lu %lu] \n", scheduled_op.cmx_address_start_,
              scheduled_op.cmx_address_end_);
        } else {
          fprintf(fptr, " resource=<none> \n");
        }
        fflush(fptr);
      }

      cmx_address_alloc(scheduled_op);
    }
    has_any_dynamic_spill_ops = false;

  }

  ///////////////////// REPACKING //////////////////////////////////////////////
  //TODO(vamsikku): get rid of repacking.
  if (passDesc.hasAttr("enable_simple_repacking"))
  {
    // Repack all ops are now original//
    std::list<scheduled_op_t> new_scheduled_ops;
    typedef mv::lp_scheduler::Repack_Input_DMA_Tasks<dag_t> repacker_t;
    typename dag_t::data_op_selector_t repackable_op_selector(input_dag);

    repacker_t repacker(input_dag, repackable_op_selector);

    repacker.repack(scheduled_ops.begin(), scheduled_ops.end(), 
        std::back_inserter(new_scheduled_ops));

    scheduled_ops = new_scheduled_ops;
    if (fptr) {
      fprintf(fptr, "[Average Repack Level]: %0.5lf\n",
            repacker.average_repack_level());
    }
  }
  //////////////////////////////////////////////////////////////////////////////



  ///////////////Save Schedule for DDR Address Generation///////////////////////
  if (passDesc.hasAttr("ddr_address_generation") &&
        passDesc.get<bool>("ddr_address_generation")) {
    typedef typename mv::lp_scheduler::Schedule_Reader_Writer<dag_t> writer_t;
    std::ostringstream schedule_state;

    bool status = writer_t::write_to_stringstream(schedule_state,
          scheduled_ops.begin(), scheduled_ops.end());
    UNUSED(status);
    assert(status);
    // save the schedule state in global params //
    auto global_params = model.getGlobalConfigParams();
    params->set<std::string>(writer_t::ddr_address_attribute(),
          schedule_state.str());
  }
  //////////////////////////////////////////////////////////////////////////////
  clock_t scheduler_algo_end_time = clock();
  double runtime = double( double(scheduler_algo_end_time) -
            double(scheduler_algo_start_time) ) / double(CLOCKS_PER_SEC);
  
  ////////////////////// Control Edge Generation ///////////////////////////////
  mv::ControlModel cmodel(model);
  control_edge_set_t control_edges(cmodel);
  bool generate_temporal_edges = !passDesc.hasAttr("no_temporal_edges");
  control_edge_generator_t algo;

  control_edges.set_zero_indegree_temporal_control(
      passDesc.hasAttr("zero_degree_temporal_edges") );

  algo.generate_control_edges(scheduled_ops.begin(), scheduled_ops.end(),
      control_edges);

  std::unordered_set<std::string> scheduled_ops_set;
  for (auto itr=scheduled_ops.begin(); itr != scheduled_ops.end(); ++itr) {
    const scheduled_op_t& op = *itr;
    scheduled_ops_set.insert( (op.op_)->getName() );
  }


  if (fptr) {
    fprintf(fptr, "\n\n");
    for (auto itr=control_edges.begin(); itr != control_edges.end(); ++itr) {
      fprintf(fptr, "control_edge: %s -> %s \n",
            (*itr).source_name(), (*itr).sink_name());
    }
  }


  { 
    control_edges.add_cmx_memory_control_edges(input_dag, model, 
        scheduled_ops.begin(), scheduled_ops.end(), generate_temporal_edges);
    printfInfo("LpScheduler:", "[Dynamic Spill Control Edge Count]: %lu\n",
        dynamic_spill_control_edges.size());

    //NOTE: dynamic_spill_control_edges for spilled CMX Concat DPU reps are 
    //included
    control_edges.add_control_edges(model, dynamic_spill_control_edges.begin(),
        dynamic_spill_control_edges.end());
  }

  ////////////////////// Control Edge Generation ///////////////////////////////


  mv::ControlModel cmodel_local(model);
  bool is_schedule_valid = cmodel_local.isDag();
  if (!is_schedule_valid) {
    if (fptr)
        fclose(fptr);
    throw lp_scheduler_exception_t("Control flow graph has cycles!");
  }

  if (fptr) {
    fprintf(fptr, "[DAG Invariant: %s]\n",
          is_schedule_valid ? "PASSED" : "FAILED");

    for (auto itr=traits_t::operations_begin(input_dag);
          itr != traits_t::operations_end(input_dag); ++itr) {
      if (scheduled_ops_set.find((*itr)->getName()) == scheduled_ops_set.end()) {
        fprintf(fptr, "[unscheduled_op]: op=%s\n", (*itr)->getName().c_str());
      }
    }
    fclose(fptr);
  }
}
