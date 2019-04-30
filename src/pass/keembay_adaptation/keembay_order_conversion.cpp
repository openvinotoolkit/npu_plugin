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
            // For HWConv wiht C < 16 set output shape to ZMajor
            dpuTask->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(dpuTask->getInputTensor(1)->getShape().ndims())));
            // We also need to set weights shape to ColMajorPlanar (see document Order.ods)
            dpuTask->getInputTensor(1)->setOrder(mv::Order(mv::Order::getColMajorPlanarID(dpuTask->getInputTensor(1)->getShape().ndims())));
            // NOTE: Anything to do about input tensor?
        }
        if((dpuTask->getOpType() == "DPUTask") && (dpuTask->get<std::string>("taskOp") == "Conv"))
        {
            // For Normal convolution, weights have to be in ColMajor order
            dpuTask->getInputTensor(1)->setOrder(mv::Order(mv::Order::getColMajorID(dpuTask->getInputTensor(1)->getShape().ndims())));
            // Anything TODO for I/O?
        }
        //Else TODO? what other cases
    }

}
