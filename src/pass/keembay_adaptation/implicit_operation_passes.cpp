#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/flow/implicit_flow.hpp"

static void resolevImplicitOperationsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(resolveImplicitOperations)
                .setFunc(resolevImplicitOperationsFcn)
                .setDescription("loops over all the candidate implicit operations and will try to add DMA to them");
    }
}

static std::map<const std::string,mv::DmaDirectionEnum> dmaDirectionStrings =
{
//    {mv::DmaDirectionEnum::CMX2DDR, "CMX2DDR"},
//    {mv::DmaDirectionEnum::DDR2CMX, "DDR2CMX"},
//    {mv::DmaDirectionEnum::CMX2UPA, "CMX2UPA"},
//    {mv::DmaDirectionEnum::UPA2CMX, "UPA2CMX"},
      {"CMX2DDR",mv::DmaDirectionEnum::CMX2DDR},
      {"DDR2CMX",mv::DmaDirectionEnum::DDR2CMX},
      {"CMX2UPA",mv::DmaDirectionEnum::CMX2UPA},
      {"UPA2CMX",mv::DmaDirectionEnum::UPA2CMX},
      {"INPUT2CMX",mv::DmaDirectionEnum::DDR2CMX},
      {"CMX2OUTPUT",mv::DmaDirectionEnum::CMX2DDR},
      {"INPUT2DDR",mv::DmaDirectionEnum::DDR2DDR},
      {"DDR2OUTPUT",mv::DmaDirectionEnum::DDR2DDR}
};

void resolevImplicitOperationsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::json::Object&)
{

    mv::OpModel om(model);

    for( auto opIt = om.opBegin(); opIt != om.opEnd(); ++ opIt)
    {
        auto opType = opIt->getOpType();
        if(opType == "Concat")
            opIt->set<mv::ImplicitFlow>("ImplicitFlow",mv::ImplicitFlow(mv::ImplicitFlow::INPUT));
        if(opType == "Slice")
            opIt->set<mv::ImplicitFlow>("ImplicitFlow",mv::ImplicitFlow(mv::ImplicitFlow::OUTPTU));

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

        //currently this phase will assume that memory locality is the only solution for implicitness.
        //TODO:: if other conditions appear, then structure them separately

        //TODO:: for now multiple output tensors is not really supported...
        // once it becomes supported, revise the logic.

        auto inputTensors = opIt->getInputTensor();
        auto outputTensor = opIt->getOutputTensor(0);

        auto outputLocation  = outputTensor->get<mv::Tensor::MemoryLocation>("Location");

        int ctr =  0;
        std::cout << "compensating for " << opIt->getName() << std::endl;
        std::vector<mv::Data::FlowSiblingIterator> flowsToRemove;

        if(implicitFlow.getCompensationDirection() == mv::ImplicitFlow::INPUT)
        {
            for (mv::Data::FlowSiblingIterator sourceFlow(opIt.leftmostInput()); sourceFlow != om.flowEnd() ; ++ sourceFlow)
            {
                auto inputTensor = sourceFlow->getTensor();
                auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");
                std::cout << "\tdoing iTensor " << inputTensor->getName() << std::endl;
                if( inputLocation != outputLocation)
                {
                    std::cout << "\t\t"<<inputTensor->getName() << "is in " << inputLocation.print() << " and " << outputTensor->getName() << " is in " << outputLocation.print() << std::endl;

                    //TODO:: PRONE TO ERRORS! corelate with Class Direction
                    const std::string directionString = inputLocation.print() + "2" + outputLocation.print();

                    std::cout << "\t\tadding DMA of " << directionString << std::endl;
                    auto compensatorOutput = om.dMATask(inputTensor,
                                                    dmaDirectionStrings[directionString],
                                                    {{},{},{},{}},
                                                    opIt->getName() + "_copy" + std::to_string(ctr));

                    compensatorOutput->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
                    auto sinkIdx = sourceFlow->get<std::size_t>("sourceOutput");
                    std::cout << "\t\tsetting " << compensatorOutput->getName() << " to " << outputLocation.print() << std::endl;
                    std::cout << "\t\tgot sinkIdx " << sinkIdx << std::endl;

                    opIt->setInputTensor(compensatorOutput,sinkIdx);
                    om.defineFlow(compensatorOutput ,opIt,sinkIdx );
                    flowsToRemove.push_back(sourceFlow);
                    ctr++;
                }
            }
            for ( int flowIdx = 0; flowIdx < flowsToRemove.size() ; flowIdx++)
            {
                om.undefineFlow(flowsToRemove[flowIdx]);
            }

            opIt->get<mv::ImplicitFlow>("ImplicitFlow").resolve();

        }
        else if (implicitFlow.getCompensationDirection() == mv::ImplicitFlow::OUTPTU)
        {
            //TODO:: REVIEW :: normally if the ImplicitFlow trait of the tensor is to compensate on output
            // it should not have multiple inputs;

            auto inputTensors = opIt->getInputTensor();

            if( inputTensors.size() > 1)
                throw mv::AttributeError("resolevImplicitOperationsFcn", " tensor " + opIt->getName() +
                                            " of type " + opIt->getOpType() +
                                            " has multiple inputs but has Implicit Compensation set to OUTPUT");

            auto inputTensor = inputTensors[0];
            auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");

            if( inputLocation != outputLocation)
            {
                std::cout << "\tdoing iTensor " << inputTensor->getName() << std::endl;
                std::cout << "\t\t"<<inputTensor->getName() << "is in " << inputLocation.print() << " and " << outputTensor->getName() << " is in " << outputLocation.print() << std::endl;


                const std::string directionString = inputLocation.print() + "2" + outputLocation.print();

                std::vector<mv::Data::OpListIterator> opsToLink;
                std::vector<std::size_t> inputSlots;
                for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
                {
                    opsToLink.push_back(sinkFlow.sink());
                    inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
                    flowsToRemove.push_back(sinkFlow);
                    std::cout << "\t\tgot sinkIdx " << sinkFlow->get<std::size_t>("sinkInput") << std::endl;
                }

                auto compensatorOutput = om.dMATask(outputTensor,
                                                        dmaDirectionStrings[directionString],
                                                        {{},{},{},{}},
                                                        opIt->getName() + "_copy" + std::to_string(ctr));

                std::cout << "\t\tsetting " << compensatorOutput->getName() << " to " << outputLocation.print() << std::endl;

                compensatorOutput->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
                outputTensor->set<mv::Tensor::MemoryLocation>("Location",inputLocation);

                for ( int flowIdx = 0; flowIdx < flowsToRemove.size() ; flowIdx++)
                {
                    om.undefineFlow(flowsToRemove[flowIdx]);
                }
                for( unsigned op = 0 ; op < opsToLink.size(); ++op)
                {
                    std::cout << "\t\tsetting "<< opsToLink[op]->getName() << "to have input " << compensatorOutput->getName() << std::endl;
                    opsToLink[op]->setInputTensor(compensatorOutput,inputSlots[op]);
                    om.defineFlow(compensatorOutput,opsToLink[op],inputSlots[op]);
                }

            }
            opIt->get<mv::ImplicitFlow>("ImplicitFlow").resolve();
        }
    }

}
