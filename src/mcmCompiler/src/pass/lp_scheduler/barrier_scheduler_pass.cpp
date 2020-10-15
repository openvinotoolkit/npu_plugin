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
  size_t producer_bound = 256UL;
  size_t real_physical_barriers =
    (size_t) passDesc.get<int>("real_physical_barriers");
  size_t barrier_bound = (real_physical_barriers/2UL);

  if (passDesc.hasAttr("barrier_bound")) {
    barrier_bound = (size_t) passDesc.get<int>("barrier_bound");
  }

  if (passDesc.hasAttr("producer_bound")) {
    producer_bound = (size_t) passDesc.get<int>("producer_bound");
  }

  if (barrier_bound > (real_physical_barriers/2UL)) {
    fprintf(stderr, "[BarrierSchedulerError]: barrier_bound must be atmost"
        " twice the real barriers");
    exit(1);
  }


  bool remove_barriers_in_upa_tail =
    (passDesc.hasAttr("remove_barriers_in_upa_tail") &&
      passDesc.get<bool>("remove_barriers_in_upa_tail"));
  bool remove_redundant_wait_barriers = 
    (passDesc.hasAttr("remove_redundant_wait_barriers") &&
      passDesc.get<bool>("remove_redundant_wait_barriers"));

  mv::ControlModel cm(model);
  dynamicallyAdjustScheduleToMeetRuntimeProblems(cm, barrier_bound,
      real_physical_barriers, producer_bound, remove_barriers_in_upa_tail,
      remove_redundant_wait_barriers);

}
