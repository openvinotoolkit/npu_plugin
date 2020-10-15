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

    auto sink = om.getOutput();
    auto sink_input_cnt = sink->inputSlots();
    while (sink_input_cnt)
    {
        auto inputTensor = sink->getInputTensor(0);
        auto source = om.getSourceOp(inputTensor);
        if (source->getOpType() == "UPATask")
            source->set<bool>("trailing", true);
        else
        {
            sink_input_cnt = 0;
            break;
        }

        sink = source;
        sink_input_cnt = sink->inputSlots();
    }
}
