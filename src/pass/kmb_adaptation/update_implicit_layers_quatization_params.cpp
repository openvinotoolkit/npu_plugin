#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void updateImplicitLayersQuantizationParamsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateImplicitLayersLocationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(UpdateImplicitLayersQuantizationParams)
            .setFunc(updateImplicitLayersQuantizationParamsFcn)
            .setDescription(
                "Update Quantization Params for Implicit Layers output after input layers have been quantized");
    }
    namespace pass
    {
        MV_REGISTER_PASS(UpdateImplicitLayersLocation)
            .setFunc(updateImplicitLayersLocationParamsFcn)
            .setDescription(
                "Update Location of Implicit Ops");
    }
}

void updateImplicitLayersQuantizationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto sortedOps = om.topologicalSort();
    for(auto opIt : sortedOps)
    {
        if (opIt->isImplicit())
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            if (input->hasAttr("quantParams"))
            {
                mv::QuantizationParams &inputQuantization = input->get<mv::QuantizationParams>("quantParams");
                opIt->set<mv::QuantizationParams>("quantParams", inputQuantization);
                output->set<mv::QuantizationParams>("quantParams", inputQuantization);
            }
        }
    }
}

std::vector<mv::Data::OpListIterator> findOutputNodeParentImplicitOps(mv::ComputationModel& model, const mv::Data::OpListIterator &op)
{
    mv::OpModel om(model);
    std::vector<mv::Data::OpListIterator> implicitOps;
    auto parentOp = om.getSourceOp(op->getInputTensor(0));
    
    while(parentOp->isImplicit())
    {
        implicitOps.push_back(parentOp); 
        parentOp = om.getSourceOp(parentOp->getInputTensor(0));
       
    }
    implicitOps.pop_back();
    return implicitOps;
}

void updateImplicitLayersLocationParamsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto sortedOps = om.topologicalSort();
    for(auto opIt : sortedOps)
    {
         std::string opType = opIt->getOpType();

        if (opType == "Slice")
        {
            auto inputMemoryLocation = opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", inputMemoryLocation);
        }
        else if (opType == "Crop")
        {
            // Recursively search for non-implicit output op
            auto outputOp = opIt.leftmostOutput().sink();

            // Crop is correctly accounted for with a spilling DMA
            if (outputOp->getOpType() == "DMATask" &&
                outputOp->get<mv::DmaDirection>("direction") == mv::NNCMX2DDR)
                continue;

            while(outputOp->isImplicit())
            {
                outputOp = outputOp.leftmostOutput().sink();
            }

            auto outputOpMemoryLocation = outputOp->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto newMemoryLocation = (outputOpMemoryLocation == mv::Tensor::MemoryLocation::OUTPUT)
                    ? mv::Tensor::MemoryLocation::OUTPUT
                    : mv::Tensor::MemoryLocation::DDR;
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", newMemoryLocation);
        }
        //NOTE: Temporary handle for the scheduler in order to place the required DMA-s for the copy operation
        else if (opType == "Copy")
        {
            if (opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::NNCMX)
            {
                auto parentOp = om.getSourceOp(opIt->getInputTensor(0));
                //Sink Ops of the Input of the Copy layer
                // auto sinkOps = findSinkLayers(dm, opIt->getInputTensor(0));
                std::vector<mv::Data::OpListIterator> sinkOps;
                auto outputFlow = parentOp.leftmostOutput();
                std::size_t copyId, dpuId = 0;
                std::unordered_map<std::string, std::vector<mv::Data::FlowSiblingIterator>> tasks_flows;
                while (outputFlow != om.flowEnd())
                {
                    if (outputFlow.sink()->getOpType() == "Copy"
                        and outputFlow.sink()->getName() == opIt->getName()) // In case of parallel branches, just this guy
                    {
                        copyId = outputFlow->get<std::size_t>("sinkInput");
                        tasks_flows["Copy"].push_back(outputFlow);
                        sinkOps.push_back(outputFlow.sink());
                    }
                    else
                    {
                        auto sinkInput = outputFlow.sink().leftmostInput();
                        while(sinkInput != om.flowEnd()){
                            if(sinkInput.source()->getName() == opIt->getName()){
                                dpuId = outputFlow->get<std::size_t>("sinkInput");
                                tasks_flows["DPUTask"].push_back(outputFlow);
                                sinkOps.push_back(outputFlow.sink());
                            }
                            ++sinkInput;
                        }
                    }
                    ++outputFlow;
                }
                auto compensatorOutput = om.dMATask(opIt->getInputTensor(0),
                                                        mv::DmaDirectionEnum::NNCMX2DDR,
                                                        0,
                                                        opIt->getName() + "_copyDMA");
                compensatorOutput->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::DDR);
                om.getSourceOp(compensatorOutput)->set<unsigned>("opId", opIt->get<unsigned>("opId"));

                for (auto sinkOp:sinkOps)
                {
                    if (sinkOp->getOpType() == "Copy")
                    {
                        //NOTE: neutral Copy has only 0
                        sinkOp->setInputTensor(compensatorOutput,0, false);
                        om.undefineFlow(tasks_flows["Copy"][0]);
                        om.defineFlow(compensatorOutput, sinkOp, copyId);
                    }
                    else
                    {
                        sinkOp->setInputTensor(compensatorOutput, 0, false);
                        om.undefineFlow(tasks_flows["DPUTask"][0]);
                        om.defineFlow(compensatorOutput, sinkOp, dpuId);
                    }
                }

            }
        }
        else if (opType == "ImplicitReshape" || opType == "ImplicitPermute")
        {
            auto input = opIt->getInputTensor(0);
            // Recursively search for non-implicit input op
            auto inputOp = om.getSourceOp(input);
            while(inputOp->isImplicit())
            {
                input = inputOp->getInputTensor(0);
                inputOp = om.getSourceOp(input);
            }
            // Recursively search for non-implicit output op
            auto outputOp = opIt.leftmostOutput().sink();
            while(outputOp->isImplicit())
            {
                outputOp = outputOp.leftmostOutput().sink();
            }
            // If to/from DDR, pick DDR as location; else use inputTensor location
            auto inputOpMemoryLocation = inputOp->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto outputOpMemoryLocation = outputOp->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto newMemoryLocation = ((inputOpMemoryLocation == mv::Tensor::MemoryLocation::DDR) ||
                                       outputOpMemoryLocation == mv::Tensor::MemoryLocation::DDR)
                    ? mv::Tensor::MemoryLocation::DDR
                    : opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", newMemoryLocation);
        }

        else if (opType == "Output" && om.getSourceOp(opIt->getInputTensor(0))->isImplicit())
        {
            auto output = om.getOutput();
            auto implicitOps = findOutputNodeParentImplicitOps(om, output);
            if(implicitOps.size())
            {
                for(auto const& implicitOp:implicitOps)
                {
                    for( size_t input = 0; input < implicitOp->inputSlots(); input++)
                        implicitOp->getInputTensor(input)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
            
                    for( size_t output = 0; output < implicitOp->outputSlots(); output++)
                        implicitOp->getOutputTensor(output)->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
                }
            }
        }
    }
}

