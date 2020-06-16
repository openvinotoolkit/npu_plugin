#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void setDpuTasksMemoryLocationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void setUPATasksMemoryLocationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(SetDpuTasksMemoryLocation)
            .setFunc(setDpuTasksMemoryLocationFcn)
            .setDescription(
                "Set Dpu Task memory location and adds copy ops if needed");

        MV_REGISTER_PASS(SetUPATasksMemoryLocation)
            .setFunc(setUPATasksMemoryLocationFcn)
            .setDescription(
                "Set UPA Task memory location and adds copy ops if needed");
    }
}

void setDpuTasksMemoryLocationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto opIt = om.opBegin();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "DPUTask")
        {
            auto taskOp = opIt->get<std::string>("taskOp");
            bool isElementWise = taskOp == "Eltwise";

            if (taskOp == "DepthwiseConv"  ||
                taskOp == "MaxPool" || taskOp == "Conv" || isElementWise || taskOp == "ChannelMajorConvolution")
            {
                auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

                if(outputMemoryLocation != mv::Tensor::MemoryLocation::NNCMX)
                {
                    auto sink = opIt.leftmostOutput().sink();
                    std::string opTypeAfter = sink->getOpType();

                    auto opItBackup = opIt;
                    if (opTypeAfter == "Crop")
                    {
                        opIt = sink;
                    }
                    auto output = opIt->getOutputTensor(0);
                    auto outputDataFlows = mv::getOutputDataFlow(om, opIt, false);

                    std::vector<mv::Data::FlowListIterator> flows;
                    for(auto outputFlow = opIt.leftmostOutput(); outputFlow != om.flowEnd(); ++outputFlow)
                        flows.push_back(outputFlow);

                    for (auto flow : flows)
                        om.undefineFlow(flow);

                    output->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
                    if (opTypeAfter == "Crop")
                        opItBackup->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);

                    mv::QuantizationParams outputQuantParams = {{},{},{},{}};
                    if (output->hasAttr("quantParams"))
                        outputQuantParams = output->get<mv::QuantizationParams>("quantParams");

                    std::string memoryLocation = outputMemoryLocation.toString();
                    if(memoryLocation == "OUTPUT" || memoryLocation == "INPUT")
                        memoryLocation = "DDR";
                    std::string stringDirection("NNCMX2"+memoryLocation);
                    mv::DmaDirection direction(stringDirection);
                    auto dpuCopyOut = om.dMATask(output, direction,opIt->getName() + "_copyOut");
                    auto dpuCopyOutOp = om.getSourceOp(dpuCopyOut);
                    dpuCopyOutOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                    if (output->hasAttr("quantParams"))
                        dpuCopyOutOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").quantize(outputQuantParams.getShift(), outputQuantParams.getMult());

                    setOutputDataFlow(om, dpuCopyOut, outputDataFlows);
                    dpuCopyOut->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

                    opIt = opItBackup;
                }

                size_t numInputs = 1;
                if (isElementWise)
                    numInputs++;

                for (size_t i = 0; i < numInputs; i++)
                {
                    auto inputMemoryLocation = opIt->getInputTensor(i)->get<mv::Tensor::MemoryLocation>("Location");
                    if(inputMemoryLocation != mv::Tensor::MemoryLocation::NNCMX ||(opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv")))
                    {
                        auto input = opIt->getInputTensor(i);
                        mv::QuantizationParams inputQuantParams = {{},{},{},{}};
                        if(input->hasAttr("quantParams"))
                            inputQuantParams = input->get<mv::QuantizationParams>("quantParams");



                        if(opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv"))
                        {
                            if (om.getSourceOp(opIt->getInputTensor(0))->getOpType() == "Slice")
                            {
                                auto slice = om.getSourceOp(opIt->getInputTensor(0));
                                auto sliceInput  = slice->getInputTensor(0);
                                auto sliceInputMemoryLocation = sliceInput->get<mv::Tensor::MemoryLocation>("Location");
                                std::string memoryLocation = sliceInputMemoryLocation.toString();
                                if(memoryLocation == "OUTPUT" || memoryLocation == "INPUT" || memoryLocation == "DEFAULT")
                                    memoryLocation = "DDR";
                                else
                                    break;
                                std::string stringDirection(memoryLocation+"2NNCMX");
                                mv::DmaDirection direction(stringDirection);

                                if (om.getSourceOp(sliceInput)->getOpType() != "DMATask")
                                {
                                    auto dpuCopyIn = om.dMATask(sliceInput, direction, opIt->getName() + "_copyIn_" + std::to_string(i));
                                    auto dpuCopyInOp = om.getSourceOp(dpuCopyIn);
                                    if(dpuCopyInOp->getOutputTensor(0)->hasAttr("quantParams"))
                                        dpuCopyInOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").quantize(inputQuantParams.getShift(), inputQuantParams.getMult());

                                    dpuCopyInOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));

                                    if(dpuCopyInOp->getOutputTensor(0)->hasAttr("quantParams"))
                                        dpuCopyInOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").quantize(inputQuantParams.getShift(), inputQuantParams.getMult());

                                    auto flows = sliceInput->get<std::set<std::string>>("flows");

                                    for(auto flowStr: flows)
                                    {
                                        auto backupFlow = dm.getDataFlow(flowStr);
                                        auto idx = backupFlow->get<std::size_t>("sinkInput");
                                        if (backupFlow.sink()->getOpType() != "DMATask")
                                        {
                                            auto sink = backupFlow.sink();
                                            om.undefineFlow(backupFlow);
                                            sink->setInputTensor(dpuCopyInOp->getOutputTensor()[0], idx, false);
                                            sink->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
                                            om.defineFlow(dpuCopyInOp, 0, sink, idx);
                                            //break;
                                        }
                                    }

                                    dpuCopyIn->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
                                }
                            }
                        }
                        else
                        {
                            std::string memoryLocation = inputMemoryLocation.toString();
                            if(memoryLocation == "OUTPUT" || memoryLocation == "INPUT" || memoryLocation == "DEFAULT")
                                memoryLocation = "DDR";
                            std::string stringDirection(memoryLocation+"2NNCMX");
                            mv::DmaDirection direction(stringDirection);
                            auto dpuCopyIn = om.dMATask(input, direction, opIt->getName() + "_copyIn_" + std::to_string(i));
                            auto dpuCopyInOp = om.getSourceOp(dpuCopyIn);

                            if(dpuCopyInOp->getOutputTensor(0)->hasAttr("quantParams"))
                                dpuCopyInOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").quantize(inputQuantParams.getShift(), inputQuantParams.getMult());

                            dpuCopyInOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));

                            auto flows = input->get<std::set<std::string>>("flows");

                            for(auto flowStr: flows)
                            {
                                auto backupFlow = dm.getDataFlow(flowStr);
                                auto idx = backupFlow->get<std::size_t>("sinkInput");
                                if (backupFlow.sink()->getName() == opIt->getName())
                                {   
                                    auto sink = backupFlow.sink();
                                    om.undefineFlow(backupFlow);
                                    sink->setInputTensor(dpuCopyIn, idx, false);
                                    om.defineFlow(dpuCopyInOp, 0, sink, idx);
                                    break;
                                }
                            }

                            dpuCopyIn->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
                        }
                        
                    }
                }
            }

        }
        ++opIt;
    }
}

void setUPATasksMemoryLocationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto opIt = om.opBegin();

    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "UPATask")
        {
            auto taskOp = opIt->get<std::string>("taskOp");
            if(taskOp == "Dummy")
            {
                ++opIt;
                continue;
            }

            // Recursively search for non-implicit output op
            auto outputOp = opIt.leftmostOutput().sink();
            while(outputOp->isImplicit())
            {
                outputOp = outputOp.leftmostOutput().sink();
            }
            //output of UPATask is ALWAYS in DDR
            // TODO: we can save 2 DMAs by giving the UPATask input in CMX (if previous op is DPUTask) and writing UPATask Output to CMX if next layer is
            // DPU task and it fits in CMX.
            auto outputOpMemoryLocation = outputOp->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto newMemoryLocation = (outputOpMemoryLocation == mv::Tensor::MemoryLocation::OUTPUT)
                    ? mv::Tensor::MemoryLocation::OUTPUT
                    : mv::Tensor::MemoryLocation::DDR;
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", newMemoryLocation);
        }
        ++opIt;
    }
}
