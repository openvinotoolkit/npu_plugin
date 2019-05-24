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
        if(dpuTask->getOpType() == "DPUTask")
        {
            auto taskOp = dpuTask->get<std::string>("taskOp");
            if (taskOp == "ChannelMajorConvolution")
            {
                // ChannelMajorConvolution is the only operation that requires input tensor in OUR ColMajor
                dpuTask->getInputTensor(0)->setOrder(mv::Order(mv::Order::getColMajorID(4)));

                // We also need to set weights shape to ColMajorPlanar (see document Order.ods)
                // This is probably wrong
                dpuTask->getInputTensor(1)->setOrder(mv::Order(mv::Order::getColMajorPlanarID(4)));
                dpuTask->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));

            }
            else
            {
                // All DPU tasks except ChannelMajor convolution (handled above) act with input tensor in ZMajor
                dpuTask->getInputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));

                // For Normal convolution, weights have to be in ColMajor order (see document Order.ods)
                if(taskOp == "Conv")
                {
                    mv::Order targetOrder("NWHC");
                    dpuTask->getInputTensor(1)->setOrder(targetOrder);
                }
            }
        }
    }

}
