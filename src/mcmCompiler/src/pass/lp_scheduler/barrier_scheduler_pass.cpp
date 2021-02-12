#include "pass/lp_scheduler/barrier_scheduler_pass.hpp"

void barrierSchedulerPass(const mv::pass::PassEntry& , mv::ComputationModel& ,
    mv::TargetDescriptor&, mv::Element& , mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(BarrierSchedulerPass)
        .setFunc(barrierSchedulerPass)
        .setDescription(
            "Runs barrier scheduler and creates barrier tasks"
        );

    }
}

typedef mv::lp_scheduler::Control_Model_Barrier_Scheduler barrier_scheduler_t;

void dynamicallyAdjustScheduleToMeetRuntimeProblems(mv::ControlModel& cm,
    size_t start_barrier_bound, size_t real_barrier_bound,
    size_t producer_bound, bool remove_barriers_in_upa_tail=false,
    bool remove_redundant_wait_barriers=false) {
  bool success = false;

  mv::lp_scheduler::Save_Restore_Control_Model save_restore(cm);

  save_restore.save();
  for (size_t barrier_bound=start_barrier_bound;
      !success && (barrier_bound>=1UL); --barrier_bound) {

    barrier_scheduler_t barrier_scheduler(cm, barrier_bound, producer_bound);
    barrier_scheduler.schedule();
    success =
        mv::lp_scheduler::Control_Model_Barrier_Checker::check_schedule(cm,
              real_barrier_bound);
    printfInfo("BarrierScheduler", "[BarrierSimulatorCheckPass(%lu)]: %s\n",
        barrier_bound, success ? "PASSED" : "FAILED"); 

    if (!success) { save_restore.restore(); }
    else {
      if (remove_redundant_wait_barriers) {
        barrier_scheduler.remove_redundant_wait_barriers();
      }
      if (remove_barriers_in_upa_tail) {
        barrier_scheduler.remove_barriers_in_upa_chain_connected_to_output();
      }
    }
  } 

  if (!success) {
    fprintf(stderr, "[Unable to schedule with two barriers] "
          "may be the graph is disconnected!");
    exit(1);
  }
  // we now have a valid barrier schedule remove tailing
}

void barrierSchedulerPass(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor&,
    mv::Element& passDesc, mv::Element&) {

  // For SC there are 8 barriers and for MC there are 32 barriers //
  std::map<std::string, size_t> barrier_config =
    {{"real_physical_barriers", 0}, {"barrier_bound", 0}, {"producer_bound", 256UL}};
  std::map<std::string, bool> barrier_remove =
    {{"remove_barriers_in_upa_tail", false}, {"remove_redundant_wait_barriers", false}};

  for (auto &attr : barrier_config)
  {
    if (model.hasGlobalConfigParam(attr.first))
      attr.second = (size_t) model.getGlobalConfigParam(attr.first).get<int>();
    else if (passDesc.hasAttr(attr.first))
      attr.second = (size_t) passDesc.get<int>(attr.first);
  }

  if (barrier_config["barrier_bound"] == 0)
    barrier_config["barrier_bound"] = (barrier_config["real_physical_barriers"]/2UL);

  if (barrier_config["barrier_bound"] > (barrier_config["real_physical_barriers"]/2UL)) {
    fprintf(stderr, "[BarrierSchedulerError]: barrier_bound must be atmost"
        " twice the real barriers");
    exit(1);
  }
  
  // In case of Profiling disable the barriers optimizations //
  std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
  if (!(globalParams->hasAttr("PerformanceCounting") && globalParams->get("PerformanceCounting"))) {
    for (auto &attr : barrier_remove)
    {
      if (model.hasGlobalConfigParam(attr.first))
        attr.second = (size_t) model.getGlobalConfigParam(attr.first).get<bool>();
      else if (passDesc.hasAttr(attr.first))
        attr.second = (size_t) passDesc.get<bool>(attr.first);
    }
  }

  mv::ControlModel cm(model);
  dynamicallyAdjustScheduleToMeetRuntimeProblems(cm, barrier_config["barrier_bound"],
      barrier_config["real_physical_barriers"], barrier_config["producer_bound"], 
      barrier_remove["remove_barriers_in_upa_tail"], barrier_remove["remove_redundant_wait_barriers"]);
}
