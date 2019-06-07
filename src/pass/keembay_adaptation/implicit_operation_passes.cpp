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
        for (mv::Data::FlowSiblingIterator sourceFlow(opIt.leftmostInput()); sourceFlow != om.flowEnd() ; ++ sourceFlow)
        {
            auto inputTensor = sourceFlow->getTensor();
            std::cout << "doing iTensor " << inputTensor->getName() << std::endl;
            auto inputLocation = inputTensor->get<mv::Tensor::MemoryLocation>("Location");

            if( inputLocation != outputLocation)
            {
                std::cout << inputTensor->getName() << "is in " << inputLocation.print() << " and " << outputTensor->getName() << " is in " << outputLocation.print() << std::endl;
                mv::Data::TensorIterator compensatorInput;
                mv::Data::TensorIterator compensatorOutput;
                mv::Tensor::MemoryLocation newLocation;

                if(implicitFlow.getCompensationDirection() == mv::ImplicitFlow::INPUT)
                {
                    compensatorInput = inputTensor;
                }
                else if (implicitFlow.getCompensationDirection() == mv::ImplicitFlow::OUTPTU)
                {
                    compensatorInput = outputTensor;
                }

                //TODO:: PRONE TO ERRORS! corelate with Direction
                const std::string direction = inputLocation.print() + "2" + outputLocation.print();
                compensatorOutput = om.dMATask(compensatorInput,
                                                mv::DmaDirection(direction),
                                                {{},{},{},{}},
                                                opIt->getName() + "_copy" + std::to_string(ctr));

                if(implicitFlow.getCompensationDirection() == mv::ImplicitFlow::INPUT)
                {
                    compensatorOutput->set<mv::Tensor::MemoryLocation>("Location",outputLocation);
                    auto sinkIdx = sourceFlow->get<std::size_t>("sourceOutput");
                    std::cout << "got sinkIdx " <<sinkIdx << std::endl;

                    opIt->setInputTensor(compensatorOutput,sinkIdx);
                    om.defineFlow(compensatorOutput ,opIt,sinkIdx );
                }
                else if (implicitFlow.getCompensationDirection() == mv::ImplicitFlow::OUTPTU)
                {
                    compensatorInput->set<mv::Tensor::MemoryLocation>("Location",inputLocation);
                    //TODO:: here, all the child ops of the slice, need to have their inputTensor
                    // set to be the compensationOutput tensor (the output of the DMA )
                }

                ctr++;

            }
        }
    }

}
