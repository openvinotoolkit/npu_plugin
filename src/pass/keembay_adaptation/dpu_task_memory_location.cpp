#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void setDpuTasksMemoryLocationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(SetDpuTasksMemoryLocation)
            .setFunc(setDpuTasksMemoryLocationFcn)
            .setDescription(
                "Set Dpu Task memory location and adds copy ops if needed");
    }
}

void setDpuTasksMemoryLocationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "DPUTask")
        {
            auto taskOp = opIt->get<std::string>("taskOp");
            if (taskOp == "ChannelMajorConvolution" ||
                taskOp == "DepthwiseConv"  ||
                taskOp == "MaxPool")
            {
                auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
                auto inputMemoryLocation = opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

                if(outputMemoryLocation != mv::Tensor::MemoryLocation::CMX)
                {
                    auto output = opIt->getOutputTensor(0);
                    auto outputOrder = output->getOrder().toString();

                    auto outputDataFlows = mv::getOutputDataFlow(om, opIt, false);
                    for(auto outputFlow = opIt.leftmostOutput(); outputFlow != om.flowEnd(); ++outputFlow)
                    {
                        om.undefineFlow(outputFlow);
                    }
                    output->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::CMX);
                    auto dpuCopyOut = om.copy(output, output->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "_copyOut");
                    om.getSourceOp(dpuCopyOut)->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                    setOutputDataFlow(om, dpuCopyOut, outputDataFlows);
                    dpuCopyOut->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
                }

                if(inputMemoryLocation != mv::Tensor::MemoryLocation::CMX)
                {
                    auto input = opIt->getInputTensor(0);
                    auto dpuCopyIn = om.copy(input, input->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "_copyIn");

                    om.getSourceOp(dpuCopyIn)->set<unsigned>("opId", opIt->get<unsigned>("opId"));

                    auto inputFlow  = opIt.leftmostInput();

                    opIt->setInputTensor(dpuCopyIn, 0, false);
                    om.undefineFlow(inputFlow);
                    om.defineFlow(dpuCopyIn, opIt, 0);
                    dpuCopyIn->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::CMX);
                }
            }

        }
        ++opIt;
    }
}
