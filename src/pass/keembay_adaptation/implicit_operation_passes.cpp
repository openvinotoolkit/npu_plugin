#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/flow/implicit_flow.hpp"

static void resolvevImplicitOperationsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(resolveImplicitOperations)
                .setFunc(resolvevImplicitOperationsFcn)
                .setDescription("loops over all the candidate implicit operations and will try to add DMA to them");
    }
}


//TODO:: unify all these enums.....
static std::map<const std::string,mv::DmaDirectionEnum> dmaDirectionStrings =
{
      {"CMX2DDR",mv::DmaDirectionEnum::CMX2DDR},
      {"DDR2CMX",mv::DmaDirectionEnum::DDR2CMX},
      {"CMX2UPA",mv::DmaDirectionEnum::CMX2UPA},
      {"UPA2CMX",mv::DmaDirectionEnum::UPA2CMX},
      {"INPUT2CMX",mv::DmaDirectionEnum::DDR2CMX},
      {"CMX2OUTPUT",mv::DmaDirectionEnum::CMX2DDR},
      {"INPUT2DDR",mv::DmaDirectionEnum::DDR2DDR},
      {"DDR2OUTPUT",mv::DmaDirectionEnum::DDR2DDR}
};

void resolvevImplicitOperationsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{
    mv::OpModel om(model);

    for( auto opIt = om.opBegin(); opIt != om.opEnd(); ++ opIt)
    {

        //TODO::the following attributes need to come either from JSON config or from OP definition
        {
            auto opType = opIt->getOpType();
            if(opType == "Concat")
                opIt->set<mv::ImplicitFlow>("ImplicitFlow",mv::ImplicitFlow(mv::ImplicitFlow::INPUT_IN_OUTPUT));
            if(opType == "Slice")
                opIt->set<mv::ImplicitFlow>("ImplicitFlow",mv::ImplicitFlow(mv::ImplicitFlow::OUTPUT_IN_INPUT));
        }

        if( !opIt->hasAttr("ImplicitFlow") )
            continue;

        auto implicitFlow = opIt->get<mv::ImplicitFlow>("ImplicitFlow");

        // The attribute is defined at leyer definition... But someone (like JSON)
        // may override the decision if it get's a candidate or not.
        // If it's not a candidate skip
        if( !implicitFlow.isCandidate())
             continue;

        if( implicitFlow.isImplicit())
        {
            //From design perspective, no pass should decide implictness before this pass
            pass.log(mv::Logger::MessageType::Warning, "found OP " + opIt->getName() + " already decided as implicit. Skipping");
            continue;
        }

        pass.log(mv::Logger::MessageType::Info,"Solving: # " + opIt->getName() + " #");

        //currently this phase will assume that memory locality is the only solution for implicitness.
        //TODO:: if other conditions appear, then structure them separately

        //TODO:: for now multiple output tensors is not really supported...
        // once it becomes supported, revise the logic.

        auto inputTensors = opIt->getInputTensor();
        auto outputTensor = opIt->getOutputTensor(0);

        auto outputLocation  = outputTensor->get<mv::Tensor::MemoryLocation>("Location");
        auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

        int ctr =  0;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;
        std::vector<mv::Data::TensorIterator> compensationOutputs;
        std::vector<std::size_t> sinkIndexes;

        if(implicitFlow.getCompensationDirection() == mv::ImplicitFlow::INPUT_IN_OUTPUT)
        {
            auto sourceFlowStart = opIt.leftmostInput();
            for (mv::Data::FlowSiblingIterator sourceFlow(sourceFlowStart); sourceFlow != om.flowEnd() ; ++ sourceFlow)
            {
                auto inputTensor = sourceFlow->getTensor();
                auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");

                if( inputLocation != outputLocation)
                {
                    //TODO:: QUant params inherited for concat
                    //TODO:: PRONE TO ERRORS! correlate with Class Direction
                    const std::string directionString = inputLocation.toString() + "2" + outputLocation.toString();
                    auto compensatorOutput = om.dMATask(inputTensor,
                                                    dmaDirectionStrings[directionString],
                                                    quantParams ,
                                                    opIt->getName() + "_copy" + std::to_string(ctr));

                    compensatorOutput->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
                    auto sinkIdx = sourceFlow->get<std::size_t>("sinkInput");

                    pass.log(mv::Logger::MessageType::Info, "Adding new DMA OP: # " + compensatorOutput->getName()+
                                                                " # after tensor: # " + inputTensor->getName() + " #");
                    flowsToRemove.push_back(sourceFlow);
                    compensationOutputs.push_back(compensatorOutput);
                    sinkIndexes.push_back(sinkIdx);

                    ctr++;
                }
            }
            for ( unsigned flowIdx = 0; flowIdx < flowsToRemove.size() ; flowIdx++)
            {
                pass.log(mv::Logger::MessageType::Info,"Setting # " + compensationOutputs[flowIdx]->getName() +
                                                        " # as input at slotIdx: " + std::to_string(sinkIndexes[flowIdx]));
                opIt->setInputTensor(compensationOutputs[flowIdx],sinkIndexes[flowIdx]);
                om.undefineFlow(flowsToRemove[flowIdx]);
                om.defineFlow(compensationOutputs[flowIdx],opIt,sinkIndexes[flowIdx]);
            }

            opIt->get<mv::ImplicitFlow>("ImplicitFlow").resolve();

        }
        else if (implicitFlow.getCompensationDirection() == mv::ImplicitFlow::OUTPUT_IN_INPUT)
        {
            //TODO:: REVIEW :: normally if the ImplicitFlow trait of the tensor is to compensate on output
            // it should not have multiple inputs;

            if( inputTensors.size() > 1)
                throw mv::AttributeError("resolevImplicitOperationsFcn", " tensor " + opIt->getName() +
                                            " of type " + opIt->getOpType() +
                                            " has multiple inputs but has Implicit Compensation set to OUTPUT");

            auto inputTensor = inputTensors[0];
            auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");

            if( inputLocation != outputLocation)
            {

                const std::string directionString = inputLocation.toString() + "2" + outputLocation.toString();

                std::vector<mv::Data::OpListIterator> opsToLink;
                std::vector<std::size_t> inputSlots;
                auto sourceFlowStart = opIt.leftmostOutput();

                for (mv::Data::FlowSiblingIterator sinkFlow(sourceFlowStart); sinkFlow != om.flowEnd(); ++sinkFlow)
                {
                    opsToLink.push_back(sinkFlow.sink());
                    inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
                    flowsToRemove.push_back(sinkFlow);
                }

                auto compensatorOutput = om.dMATask(outputTensor,
                                                        dmaDirectionStrings[directionString],
                                                        quantParams,
                                                        opIt->getName() + "_copy" + std::to_string(ctr));

                pass.log(mv::Logger::MessageType::Info,"Adding new DMA OP: # " + compensatorOutput->getName() +
                                                            " as output to # " + opIt->getName());

                // TODO: NOTE: I should be using the relocate function.... but that may fail in case the "output" of slice is forced.
                //Technically, for compensation I should not be adding "new dma" to the slice "outputTensor" but rather
                //but a new tensor as the Slice output and DMA between them. But the API is not really friendly for this
                //case so the "forced" attribute will be inherited by the output of the DMA itself....

                compensatorOutput->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
                if(outputTensor->get<mv::Tensor::MemoryLocation>("Location").isForced())
                    compensatorOutput->get<mv::Tensor::MemoryLocation>("Location").force();

                outputTensor->set<mv::Tensor::MemoryLocation>("Location",inputLocation);

                for ( unsigned flowIdx = 0; flowIdx < flowsToRemove.size() ; flowIdx++)
                {
                    om.undefineFlow(flowsToRemove[flowIdx]);
                }
                for( unsigned op = 0 ; op < opsToLink.size(); ++op)
                {
                    pass.log(mv::Logger::MessageType::Info," Setting # " + compensatorOutput->getName() +
                                                                "# as input to: # " + opsToLink[op]->getName() +
                                                                "# at slotIdx: " + std::to_string(inputSlots[op]));

                    opsToLink[op]->setInputTensor(compensatorOutput,inputSlots[op]);
                    om.defineFlow(compensatorOutput,opsToLink[op],inputSlots[op]);
                }

            }
            opIt->get<mv::ImplicitFlow>("ImplicitFlow").resolve();
        }
    }
}
