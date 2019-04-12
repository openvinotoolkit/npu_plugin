#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void kmbOrderConversion(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(KMBOrderConversion)
            .setFunc(kmbOrderConversion)
            .setDescription(
                "Pass converts the order of the output when required in KMB");
    }
}

void kmbOrderConversion(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if((dpuTask->getOpType() == "DPUTask") && (dpuTask->get<std::string>("taskOp") == "ChannelMajorConvolution"))
        {
            //For HWConv wiht C < 16 set output shape to ZMajor
            dpuTask->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getRowMajorPlanarID(dpuTask->getOutputTensor(0)->getShape().ndims())));
        }
        //Else TODO? what other cases
    }

}
