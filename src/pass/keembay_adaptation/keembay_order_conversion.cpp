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
            // NOTE: Keeping this structure while waiting confirmation from architecture team
            if (dpuTask->get<std::string>("taskOp") == "ChannelMajorConvolution")
            {
                // For HWConv wiht C < 16 set output shape to ZMajor and Input shape to ColumnMajor
                dpuTask->getInputTensor(0)->setOrder(mv::Order(mv::Order::getColMajorID(4)));
                dpuTask->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));

                // We also need to set weights shape to ColMajorPlanar (see document Order.ods)
                dpuTask->getInputTensor(1)->setOrder(mv::Order(mv::Order::getColMajorPlanarID(4)));
            }
            else if(dpuTask->get<std::string>("taskOp") == "Conv")
            {
                // For HWConv wiht C < 16 set output shape to ZMajor
                dpuTask->getInputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));
                dpuTask->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));

                // For Normal convolution, weights have to be in ColMajor order (see document Order.ods)
                dpuTask->getInputTensor(1)->setOrder(mv::Order(mv::Order::getColMajorID(4)));
            }
            else
            {
                dpuTask->getInputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));
                dpuTask->getOutputTensor(0)->setOrder(mv::Order(mv::Order::getZMajorID(4)));
            }
        }
    }

}
