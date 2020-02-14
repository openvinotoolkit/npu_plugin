#include "include/mcm/pass/pass_utils.hpp"

std::vector<std::pair<mv::Data::OpListIterator,size_t>> mv::getOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, bool deleteOp)
{
    std::vector<std::pair<mv::Data::OpListIterator,size_t>> toReturn;

    for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
    {
        auto consumer = output.sink();
        auto slot = output->get<size_t>("sinkInput");
        toReturn.push_back(std::make_pair(consumer, slot));
    }

    if(deleteOp)
    {
        auto backup = opIt;
        ++opIt;
        om.removeOp(backup);
    }

    return toReturn;
}

void mv::setOutputDataFlow(mv::OpModel& om, mv::Data::TensorIterator &dpuTaskOutputTensor, const std::vector<std::pair<mv::Data::OpListIterator,size_t>>& outDataFlows)
{
    for(auto& flowPair: outDataFlows)
    {
        flowPair.first->setInputTensor(dpuTaskOutputTensor, flowPair.second, false);
        om.defineFlow(dpuTaskOutputTensor, flowPair.first, flowPair.second);
    }
}

std::vector<mv::Control::OpListIterator> mv::getOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator opIt)
{
    std::vector<mv::Control::OpListIterator> toReturn;

    for(auto outputFlow = opIt.leftmostChild(); outputFlow != cm.opEnd(); ++outputFlow)
        toReturn.push_back(outputFlow);
    return toReturn;
}

std::vector<mv::Control::OpListIterator> mv::getInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator opIt)
{
    std::vector<mv::Control::OpListIterator> toReturn;

    for(auto outputFlow = opIt.leftmostParent(); outputFlow != cm.opEnd(); ++outputFlow)
        toReturn.push_back(outputFlow);
    return toReturn;
}

void mv::setInputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& inputControlFlows)
{
    for(auto& inputOp: inputControlFlows)
        cm.defineFlow(inputOp, op);
}

void mv::setOutputControlFlow(mv::ControlModel& cm, mv::Control::OpListIterator op, const std::vector<mv::Control::OpListIterator>& outputControlFlows)
{
    for(auto& outputOp: outputControlFlows)
        cm.defineFlow(op, outputOp);
}

mv::Data::OpListIterator mv::linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt,
                                                 mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (auto sinkFlow = opIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt"
            || paramOp->getOpType() == "ConstantDataElement")
        {
            auto backUp = paramOp;
            ++paramOp;
            om.removeOp(backUp);
        }
        else
            ++paramOp;
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    if(sourceTensor == om.tensorEnd())
        sourceTensor = parentOpIt->getOutputTensor(0);

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}
