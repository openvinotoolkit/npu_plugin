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
    mv::DataModel dm(model);

    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "DPUTask")
        {
            auto taskOp = opIt->get<std::string>("taskOp");
            if (taskOp == "ChannelMajorConvolution" ||
                taskOp == "DepthwiseConv"  ||
                taskOp == "MaxPool" || taskOp == "Conv")
            {
                auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
                auto inputMemoryLocation = opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

                if(outputMemoryLocation != mv::Tensor::MemoryLocation::CMX)
                {
                    auto output = opIt->getOutputTensor(0);
                    auto outputOrder = output->getOrder().toString();

                    auto outputDataFlows = mv::getOutputDataFlow(om, opIt, false);

                    std::vector<mv::Data::FlowListIterator> flows;
                    for(auto outputFlow = opIt.leftmostOutput(); outputFlow != om.flowEnd(); ++outputFlow)
                    {
                        flows.push_back(outputFlow);
                    }

                    for (auto flow : flows)
                        om.undefineFlow(flow);

                    output->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::CMX);

//                    auto dpuCopyOut = om.copy(output, output->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "_copyOut");
                    auto dpuCopyOut = om.dMATask(output,mv::DmaDirectionEnum::CMX2DDR,output->get<mv::QuantizationParams>("quantParams"),opIt->getName() + "_copyOut");

                    om.getSourceOp(dpuCopyOut)->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                    setOutputDataFlow(om, dpuCopyOut, outputDataFlows);
                    dpuCopyOut->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
                }

                if(inputMemoryLocation != mv::Tensor::MemoryLocation::CMX)
                {
                    auto input = opIt->getInputTensor(0);
//                    auto dpuCopyIn = om.copy(input, input->get<mv::QuantizationParams>("quantParams"), opIt->getName() + "_copyIn");
                    auto dpuCopyIn = om.dMATask(input,mv::DmaDirectionEnum::DDR2CMX,input->get<mv::QuantizationParams>("quantParams"),opIt->getName() + "_copyIn");
                    auto dpuCopyInOp = om.getSourceOp(dpuCopyIn);

                    om.getSourceOp(dpuCopyIn)->set<unsigned>("opId", opIt->get<unsigned>("opId"));

                    auto flows = input->get<std::set<std::string>>("flows");

                    for(auto flowStr: flows)
                    {
                        auto backupFlow = dm.getDataFlow(flowStr);
                        auto idx = backupFlow->get<std::size_t>("sinkInput");
                        if (idx == 0)
                        {
                            auto sink = backupFlow.sink();
                            om.undefineFlow(backupFlow);
                            sink->setInputTensor(dpuCopyIn, idx, false);
                            om.defineFlow(dpuCopyInOp, 0, sink, idx);
                            break;
                        }
                    }

                    dpuCopyIn->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::CMX);
                }
            }

        }
        ++opIt;
    }
}
