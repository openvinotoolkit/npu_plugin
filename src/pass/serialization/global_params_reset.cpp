#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"



static void GlobalParamsResetFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&);

namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(GlobalParamsReset)
        .setFunc(GlobalParamsResetFcn)
        .setDescription(
            "Resets any global parameters that need to be reset after serialization."
        );
    }
}

static void GlobalParamsResetFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    //Reset the global barrier params
    mv::Barrier::reset();
}

