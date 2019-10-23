#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "lp_scheduler/operation_precedence_dag.hpp"
#include "scheduler/feasible_scheduler.hpp"
#include <unordered_set>

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


void LpSchedulerPass(const mv::pass::PassEntry& pass,
    mv::ComputationModel& model, mv::TargetDescriptor& target,
    mv::Element& passDesc, mv::Element& compOutput) {

  mv::ControlModel cm(model);
  control_dag_t input_dag(cm);
  typedef mv::lp_scheduler::scheduler_traits<control_dag_t> traits_t;

  auto params = model.getGlobalConfigParams();

  control_dag_t::resource_t upper_bound = params->get<unsigned>("cmx");
  std::string output_file = passDesc.get<std::string>("output");

  FILE *fptr = fopen(output_file.c_str(), "w");
  assert(fptr);


  mv::scheduler::mv_control_lp_scheduler_t scheduler(input_dag, upper_bound ),
      end;

  size_t op_count = 0UL;
  std::unordered_set<std::string> scheduled_ops;
  size_t prev_time = std::numeric_limits<size_t>::max();
  while (scheduler != end) {
    auto rstate = scheduler.resource_state();
    auto rinfo = rstate.get_resource_usage_info(*scheduler);
    auto curr_time = scheduler.current_time();

    fprintf(fptr, "op=%s time=%lu mem_allocation=[%lu %lu] mem=%lu\n",
        (*scheduler)->getName().c_str(), scheduler.current_time(),
        rinfo.begin_, rinfo.end_, (rinfo.end_ - rinfo.begin_) );

    scheduled_ops.insert( (*scheduler)->getName() );

    ++scheduler;
    op_count++;
  }

  fprintf(fptr, "scheduled_ops = %lu\n", op_count);
  size_t total_op_count = 0UL;

  for (auto itr=traits_t::operations_begin(input_dag);
        itr != traits_t::operations_end(input_dag); ++itr) {
    if (scheduled_ops.find((*itr)->getName()) == scheduled_ops.end()) {
      fprintf(fptr, "[unscheduled] op=%s\n", (*itr)->getName().c_str());
    }
    total_op_count++;
  }

  fprintf(fptr, "expected scheduled_ops = %lu\n", total_op_count);
  fclose(fptr);
}
