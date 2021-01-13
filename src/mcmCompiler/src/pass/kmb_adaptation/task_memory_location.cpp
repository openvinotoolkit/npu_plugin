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
                    //NOTE: this is used for subdilation convolutions which are going to have only 1 sink
                    auto sinkOp = findSinkLayers(dm, opIt->getOutputTensor()[0])[0];

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

                    auto outputQuantParams = output->getQuantParams();

                    std::string memoryLocation = outputMemoryLocation.toString();
                    if(memoryLocation == "OUTPUT" || memoryLocation == "INPUT")
                        memoryLocation = "DDR";
                    std::string stringDirection("NNCMX2"+memoryLocation);
                    mv::DmaDirection direction(stringDirection);
                    auto dpuCopyOut = om.dMATask(opIt->getName() + "_copyOut", output, direction, 0);
                    if (sinkOp->hasAttr("dilatedWidthConcat") && sinkOp->get<bool>("dilatedWidthConcat"))
                    {
                        std::size_t slot = 0;
                        for (std::size_t inputConcatTensorIdx = 0; inputConcatTensorIdx < sinkOp->getInputTensor().size();
                             inputConcatTensorIdx++)
                            if (sinkOp->getInputTensor()[inputConcatTensorIdx]->getName() == output->getName())
                                slot = inputConcatTensorIdx;
                        //NOTE: only the tensor which goes to ddr, the dst should have the dilated strides
                        dpuCopyOut->set<bool>("dilatedWidthConcat", true);
                        dpuCopyOut->set<unsigned>("dilationFactor",
                                                         sinkOp->get<unsigned>("dilationFactor"));
                        dpuCopyOut->set<std::size_t>("inputConcatTensorIdx", slot);
                        dpuCopyOut->set<std::size_t>("lineofConcatHeight",
                                                    sinkOp->get<std::size_t>("lineofConcatHeight"));
                    }
                    else if ((opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv")) || (opIt->hasAttr("DeconvSubConv") && opIt->get<bool>("DeconvSubConv")))
                    {
                        //NOTE: only the tensor which goes to ddr, the dst should have the dilated strides
                        dpuCopyOut->set<bool>("dilatedWidthConcat", true);
                        dpuCopyOut->set<unsigned>("dilationFactor",
                                                         opIt->get<unsigned>("originalDilationFactor"));
                        dpuCopyOut->set<std::size_t>("inputConcatTensorIdx", opIt->get<std::vector<std::size_t>>("subConvsCoordinates")[1]);
                        dpuCopyOut->set<std::size_t>("lineofConcatHeight",
                                                    opIt->get<std::vector<std::size_t>>("subConvsCoordinates")[0]);
                        if (opIt->hasAttr("streamHId"))
                        {
                            auto streamHId = opIt->get<unsigned>("streamHId");
                            auto symmetrical_first_dimensionH = opIt->get<std::size_t>("symmetrical_first_dimensionH");
                            dpuCopyOut->set<unsigned>("streamHId", streamHId);
                            dpuCopyOut->set<std::size_t>("symmetrical_first_dimensionH",
                                                         symmetrical_first_dimensionH);
                        }
                        else if (opIt->hasAttr("streamKId"))
                        {
                            auto streamKId = opIt->get<unsigned>("streamKId");
                            auto symmetrical_first_dimensionK = opIt->get<std::size_t>("symmetrical_first_dimensionK");
                            dpuCopyOut->set<unsigned>("streamKId", streamKId);
                            dpuCopyOut->set<std::size_t>("symmetrical_first_dimensionK",
                                                         symmetrical_first_dimensionK);
                        }
                    }
                    auto dpuCopyOutOp = om.getSourceOp(dpuCopyOut);
                    dpuCopyOutOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                    if (output->hasAttr("quantParams"))
                        dpuCopyOut->get<mv::QuantizationParams>("quantParams").quantize(outputQuantParams.getShift(), outputQuantParams.getMult());

                    setOutputDataFlow(om, dpuCopyOut, outputDataFlows);
                    dpuCopyOut->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

                    opIt = opItBackup;
                }

                size_t numInputs = 1;
                if (isElementWise)
                    numInputs = opIt->inputSlots();

                for (size_t i = 0; i < numInputs; i++)
                {
                    auto input = opIt->getInputTensor(i);
                    auto inputMemoryLocation = input->get<mv::Tensor::MemoryLocation>("Location");

                    if(inputMemoryLocation != mv::Tensor::MemoryLocation::NNCMX ||(opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv")))
                    {
                        std::vector<mv::Data::FlowListIterator> flowsToRemove;
                        std::vector<std::size_t> inSlots;
                        std::vector<mv::Data::OpListIterator> flowSinks;

                        if(opIt->hasAttr("DilatedSubConv") && opIt->get<bool>("DilatedSubConv"))
                        {
                            if (om.getSourceOp(input)->getOpType() == "Slice")
                            {
                                auto slice = om.getSourceOp(input);
                                auto sliceInput  = slice->getInputTensor(0);
                                auto const& sliceInputMemoryLocation = sliceInput->get<mv::Tensor::MemoryLocation>("Location");

                                std::string memoryLocation = sliceInputMemoryLocation.toString();
                                if(memoryLocation == "OUTPUT" || memoryLocation == "INPUT" || memoryLocation == "DEFAULT")
                                    memoryLocation = "DDR";

                                if (memoryLocation != "DDR")
                                    break;

                                if (om.getSourceOp(sliceInput)->getOpType() != "DMATask")
                                {
                                    auto const& flows = sliceInput->get<std::set<std::string>>("flows");
                                    for(auto const& flowStr : flows)
                                    {
                                        auto flow = dm.getDataFlow(flowStr);
                                        if (flow.sink()->getOpType() != "DMATask")
                                        {
                                            flowsToRemove.push_back(flow);
                                            flowSinks.push_back(flow.sink());
                                            inSlots.push_back(flow->get<std::size_t>("sinkInput"));
                                        }
                                    }

                                    mv::DmaDirection const direction(mv::DmaDirectionEnum::DDR2NNCMX);
                                    auto dpuCopyIn = mv::insertDMAReplacementRemoveFlows(om, opIt, sliceInput,
                                        direction, 0, flowsToRemove, inSlots, flowSinks,
                                        opIt->getName() + "_copyIn_" + std::to_string(i));

                                    for (auto& sink : flowSinks) {
                                        sink->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
                                    }

                                    dpuCopyIn->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
                                }
                            }
                        }
                        else
                        {
                            auto const& flows = input->get<std::set<std::string>>("flows");
                            for(auto const& flowStr : flows)
                            {
                                auto flow = dm.getDataFlow(flowStr);
                                if (flow.sink()->getName() == opIt->getName())
                                {
                                    flowsToRemove.push_back(flow);
                                    flowSinks.push_back(flow.sink());
                                    inSlots.push_back(flow->get<std::size_t>("sinkInput"));
                                }
                            }

                            std::string memoryLocation = inputMemoryLocation.toString();
                            if(memoryLocation == "OUTPUT" || memoryLocation == "INPUT" || memoryLocation == "DEFAULT")
                                memoryLocation = "DDR";

                            auto dmaTaskOut = mv::insertDMAReplacementRemoveFlows(om, opIt, input,
                                mv::DmaDirection(memoryLocation + "2NNCMX"),
                                0, flowsToRemove, inSlots, flowSinks,
                                opIt->getName() + "_copyIn_" + std::to_string(i));
                            dmaTaskOut->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::NNCMX);
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

            // Adjust memory location for subsequent implicit operation on the path if the UPATask
            // output tensor location was modified to OUTPUT.
            // TODO: For now below loop is a limited implementation and will handle correctly cases with single sink. In case
            // there are branches implementation should be extended.
            if (outputOpMemoryLocation == mv::Tensor::MemoryLocation::OUTPUT)
            {
                outputOp = opIt.leftmostOutput().sink();
                while(outputOp->isImplicit())
                {
                    outputOp->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputOpMemoryLocation);
                    outputOp = outputOp.leftmostOutput().sink();
                }
            }
        }
        ++opIt;
    }
}
