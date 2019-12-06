#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
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
typedef mv::lp_scheduler::Scheduled_Op scheduled_op_t;
typedef mv::lp_scheduler::Control_Edge control_edge_t;
typedef mv::lp_scheduler::Control_Edge_Set control_edge_set_t;
typedef mv::lp_scheduler::Control_Edge_Generator<scheduled_op_t>
  control_edge_generator_t;

namespace mv {
namespace lp_scheduler {

template<>
struct interval_traits<scheduled_op_t> {
  typedef size_t unit_t;
  typedef scheduled_op_t interval_t;

  static unit_t interval_begin(const interval_t& interval) {
    return interval.cmx_address_start_;
  }

  static unit_t interval_end(const interval_t& interval) {
    return interval.cmx_address_end_;
  }

}; // struct interval_traits<Scheduled_Op> //

} // namespace lp_scheduler //
} // namespace mv //

void LpSchedulerAllocatorPass(mv::ComputationModel& model) {
  mv::lp_scheduler::Tensor_Allocator_Assignment alloc(model);
  mv::OpModel om(model);

  for (auto itr=om.getInput(); itr!=om.opEnd(); ++itr) {
    mv::Op &op = *itr;
    if (!op.outputSlots()) { continue; }

    printf("[op=%s]\n", op.getName().c_str());
    fflush(stdout);
    mv::Data::TensorIterator tensor_itr = op.getOutputTensor(0UL);
    alloc(tensor_itr);
  }
}


void LpSchedulerPass(const mv::pass::PassEntry& pass,
    mv::ComputationModel& model, mv::TargetDescriptor& target,
    mv::Element& passDesc, mv::Element& compOutput) {

  if (passDesc.hasAttr("allocator_mode")) {
    LpSchedulerAllocatorPass(model);
  }

  typedef mv::lp_scheduler::mv_memory_scheduler_with_spilling_t scheduler_t;
  typedef typename scheduler_t::scheduled_op_info_t scheduled_op_info_t;

  mv::OpModel cm(model);
  dag_t input_dag(cm);

  // short circuit Slice and Crop //
  std::vector<std::string> short_circuit_ops = {"Slice", "Crop"};
  for (auto short_circuit_itr=short_circuit_ops.begin();
      short_circuit_itr!=short_circuit_ops.end(); ++short_circuit_itr) {
    input_dag.short_circuit_all_unit_indegree_outdegree_ops_of_this_type(
        *short_circuit_itr);
  }

  typedef mv::lp_scheduler::scheduler_traits<dag_t> traits_t;
  auto params = model.getGlobalConfigParams();

  dag_t::resource_t upper_bound = params->get<unsigned>("cmx");
  std::string output_file = passDesc.get<std::string>("output");
  FILE *fptr = fopen(output_file.c_str(), "w");
  assert(fptr);


  // generate tensor addresses //
  mv::lp_scheduler::Tensor_Address_Assignment<scheduler_t>
      cmx_address_alloc(model);
  scheduler_t scheduler(input_dag, upper_bound), scheduler_end;
  std::list<scheduled_op_t> scheduled_ops;
  bool has_any_implicit_ops = false;

  while (scheduler != scheduler_end) {
    const scheduled_op_info_t &scheduled_op = *scheduler;

    if (scheduled_op.op_type_name() != std::string("ORIGINAL")) {
      has_any_implicit_ops = true;
    }

    mv::Op const *op = scheduled_op.op_;
    size_t rbegin = scheduled_op.begin_resource();
    size_t rend = scheduled_op.end_resource();

    scheduled_ops.push_back(scheduled_op_t(op, scheduled_op.time_,
          rbegin, rend));

    fprintf(fptr, "op = %-20s  type = %-15s  time = %lu ",
        (scheduled_op.op_)->getName().c_str(), scheduled_op.op_type_name(),
          scheduled_op.time_);
    fflush(fptr);

    if (scheduled_op.has_active_resource()) {
      fprintf(fptr, " resource=[%lu %lu]\n", scheduled_op.begin_resource(),
          scheduled_op.end_resource());
    } else {
      fprintf(fptr, " resource=<none>\n");
    }

    cmx_address_alloc(scheduled_op);
    ++scheduler;
  }

  mv::ControlModel cmodel(model);
  control_edge_set_t control_edges(cmodel);
  control_edge_generator_t algo;

  if (!has_any_implicit_ops) {
    algo.generate_control_edges(scheduled_ops.begin(), scheduled_ops.end(),
        control_edges);
  } else {
    fprintf(fptr, "WARNING: control edge generation with implicit ops not yet"
        " implemented\n");
  }

  std::unordered_set<std::string> scheduled_ops_set;
  for (auto itr=scheduled_ops.begin(); itr != scheduled_ops.end(); ++itr) {
    const scheduled_op_t& op = *itr;
    scheduled_ops_set.insert( (op.op_)->getName() );
  }

  fprintf(fptr, "\n\n");
  for (auto itr=control_edges.begin(); itr != control_edges.end(); ++itr) {
    fprintf(fptr, "control_edge: %s -> %s \n",
          (*itr).source_name(), (*itr).sink_name());
  }


  if (!has_any_implicit_ops) {
    control_edges.add_edges_to_fresh_control_model(input_dag, model, 
        scheduled_ops.begin(), scheduled_ops.end());
  } else {
    //TODO(vamsikku): add new operations corresponding to implicit write and 
    //implicit reads.
    fprintf(fptr, "WARNING: control edge generation with implicit ops not yet"
        " implemented\n");
  }


  {
    mv::ControlModel cmodel_local(model);
    fprintf(fptr, "[DAG Invariant: %s]\n",
          cmodel_local.isDag() ? "PASSED" : "FAILED");
  }


  for (auto itr=traits_t::operations_begin(input_dag);
        itr != traits_t::operations_end(input_dag); ++itr) {
    if (scheduled_ops_set.find((*itr)->getName()) == scheduled_ops_set.end()) {
      fprintf(fptr, "[unscheduled_op]: op=%s\n", (*itr)->getName().c_str());
    }
  }
  fclose(fptr);
}
