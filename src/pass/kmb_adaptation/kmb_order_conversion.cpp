#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void kmbOrderConversion(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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

void kmbOrderConversion(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            //handle channel major convolution (only possible if enabled in comp descriptor)
            auto taskOp = dpuTask->get<std::string>("taskOp");
            if (taskOp == "ChannelMajorConvolution")
            {
                // ChannelMajorConvolution is the only operation that requires input tensor in OUR ColMajor
                dpuTask->getInputTensor(0)->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                if (om.getSourceOp(dpuTask->getInputTensor(0))->getOpType() == "Slice")
                {
                    auto inputImplicitOp = om.getSourceOp(dpuTask->getInputTensor(0));
                    inputImplicitOp->getInputTensor(0)->setOrder(mv::Order::getColMajorID(4));
                }

                // We also need to set weights shape to ColMajor (see document Order.ods)
                mv::Order targetOrder(mv::Order::getColMajorID(4));
                dpuTask->getInputTensor(1)->setOrder(targetOrder);
                dpuTask->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));
                if (om.getSourceOp(dpuTask->getInputTensor(1))->getOpType() == "Slice")
                {
                    auto kernelImplicitOp = om.getSourceOp(dpuTask->getInputTensor(1));
                    kernelImplicitOp->getInputTensor(0)->setOrder(targetOrder);
                }
            }
            else 
            {
                dpuTask->getInputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));
                if (om.getSourceOp(dpuTask->getInputTensor(0))->getOpType() == "Slice")
                {
                    auto inputImplicitOp = om.getSourceOp(dpuTask->getInputTensor(0));
                    inputImplicitOp->getInputTensor(0)->setOrder(mv::Order::getZMajorID(4));
                }
                if(taskOp == "Conv")
                {
                    mv::Order targetOrder("NHWC");
                    dpuTask->getInputTensor(1)->setOrder(targetOrder);
                    if (om.getSourceOp(dpuTask->getInputTensor(1))->getOpType() == "Slice")
                    {
                        auto kernelImplicitOp = om.getSourceOp(dpuTask->getInputTensor(1));
                        kernelImplicitOp->getInputTensor(0)->setOrder(targetOrder);
                    }
                }
            }
        }
    }

}
