#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"


static void markTrailingUPATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(MarkTrailingUPATasks)
        .setFunc(markTrailingUPATasksFcn)
        .setDescription(
            "This pass identifies UPATasks at the end of the network that are linearly dependent, for optimized execution in runtime."
        );

    }

}

void markTrailingUPATasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto upaTasks = om.getOps("UPATask");
    for(auto& task : upaTasks)
    {
        bool is_trailing = true;
        auto nextOp = task.leftmostOutput().sink();
        while (nextOp->getOpType() != "Output")
        {
            if (nextOp->getOpType() == "DPUTask")
                is_trailing = false;

            nextOp = nextOp.leftmostOutput().sink();
        }

        task->set<bool>("trailing", is_trailing);
    }

}
