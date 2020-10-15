#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void ImplicitOutputDTypeUpdateFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ImplicitOutputDTypeUpdate)
        .setFunc(ImplicitOutputDTypeUpdateFcn)
        .setDescription(
            "This pass updates implicitOutput DTypes alignment between input and output tensors."
        );
    }
}

void ImplicitOutputDTypeUpdateFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto implicitOutputOps = om.getOps("ImplicitOutput");

    if(implicitOutputOps.size() == 0)
        return;

    for(auto& implicitOutput : implicitOutputOps)
        if(implicitOutput->getInputTensor(0)->getDType() != implicitOutput->getOutputTensor(0)->getDType())
        {
            implicitOutput->getOutputTensor(0)->setDType(implicitOutput->getInputTensor(0)->getDType());
        }
        else
        {
            continue;
        }

}
