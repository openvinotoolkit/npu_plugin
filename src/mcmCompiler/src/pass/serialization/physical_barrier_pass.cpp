#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "lp_scheduler/runtime_simulator.hpp"

static void AssignPhysicalBarrierFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td,
    mv::Element& passDesc, mv::Element&);
namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AssignPhysicalBarrier)
        .setFunc(AssignPhysicalBarrierFcn)
        .setDescription("Assigns physical barriers by runtime simulation");
    }
}

static void AssignPhysicalBarrierFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor& td,
    mv::Element& passDesc, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::ControlModel cmodel(model);
    bool isStatic= false;
    auto globalParams = model.getGlobalConfigParams();
    if (globalParams->hasAttr("enableStaticBarriers"))
    {
      isStatic= globalParams->get<bool>("enableStaticBarriers");
    }

    if (isStatic) {
      size_t real_physical_barriers = 0;
      if (model.hasGlobalConfigParam("real_physical_barriers"))
          real_physical_barriers =
          (size_t) model.getGlobalConfigParam("real_physical_barriers").get<int>();
      else if (passDesc.hasAttr("real_physical_barriers"))
          real_physical_barriers =
          (size_t) passDesc.get<int>("real_physical_barriers");
      bool success =
          mv::lp_scheduler::Control_Model_Barrier_Assigner::assign_physical_id(cmodel,
                                                                               real_physical_barriers);
      if (success) {
        mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeSimulator",
                        "Finished Runtime Simulation: Pass");
      } else {
        mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeSimulator",
                        "Finished Runtime Simulation: Fail");
        throw "Failed to pass runtime simulation\n";
      }
    }
}
