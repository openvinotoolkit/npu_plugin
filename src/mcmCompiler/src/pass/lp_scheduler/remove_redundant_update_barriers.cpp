#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "lp_scheduler/remove_redundant_update_barriers.hpp"

static void RemoveRedundantUpdateBarriers(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
    mv::Element&, mv::Element&);

namespace mv {
  namespace pass {

    MV_REGISTER_PASS(RemoveRedundantUpdateBarriers)
      .setFunc(RemoveRedundantUpdateBarriers)
      .setDescription("Remove redundant update barriers before serialization");

  } // namespace mv //
} // namespace pass //

void RemoveRedundantUpdateBarriers(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& ,
    mv::Element&) {

  mv::ControlModel cmodel(model);
  mv::lp_scheduler::Remove_Redundant_Update_Barriers redundant_barriers(cmodel);

  redundant_barriers.remove();
}
