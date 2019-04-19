#include "include/mcm/pass/pass_registry.hpp"
//#include "include/mcm/computation/model/control_model.hpp"
//#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"



static void GlobalParamsResetFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::json::Object&);

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

static void GlobalParamsResetFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& compilationDescriptor, mv::json::Object&)
{
    //Reset the global barrier params
    std::cout<<"here inside the enw pass"<<std::endl;
    mv::Barrier::reset();
}

