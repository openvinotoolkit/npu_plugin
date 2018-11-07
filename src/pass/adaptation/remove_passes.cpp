#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void removeDropOut(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(RemoveDropOut)
        .setFunc(removeDropOut)
        .setGenre(PassGenre::Adaptation)
        .setDescription(
            "Removes dropout layers from the network"
        );

    }

}

void removeDropOut(mv::ComputationModel &model, mv::TargetDescriptor &, mv::json::Object &, mv::json::Object &)
{

    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Dropout")
        {
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                std::size_t inputIdx = sinkFlow->get<std::size_t>("sinkInput");
                sinkFlow.sink()->erase("input" + std::to_string(inputIdx));
                om.defineFlow(sourceTensor, sinkFlow.sink(), inputIdx);
            }

            while (opIt.parentsSize() > 1)
            {
                auto paramOp = opIt.leftmostParent();
                ++paramOp;
                om.removeOp(paramOp);
            }

            om.removeOp(opIt);
            opIt = parentOpIt;
        }
    }
}
