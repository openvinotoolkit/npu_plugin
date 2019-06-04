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
