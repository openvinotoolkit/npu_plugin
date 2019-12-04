#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void removeDropOut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeInterpNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(RemoveDropOut)
        .setFunc(removeDropOut)
        .setDescription(
            "Removes dropout layers from the network"
        );

        MV_REGISTER_PASS(RemoveInterpNoOp)
        .setFunc(removeInterpNoOpFcn)
        .setDescription(
            "Removes Interp layers that do nothing"
        );

    }

}


mv::Data::OpListIterator linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    while(opIt.parentsSize() > 1)
    {
        auto paramOp = opIt.leftmostParent();
        ++paramOp;
        om.removeOp(paramOp);
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}
void removeDropOut(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Dropout")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
            if (outputMemoryLocation.isForced())
            {
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
        }
    }
}

void removeInterpNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->getOpType() == "Interp")
        {
            auto inputShape = opIt->getInputTensor(0)->getShape();
            auto outputShape = opIt->getOutputTensor(0)->getShape();
            if (inputShape == outputShape)
            {
                auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
                auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

                auto sourceTensor = parentOpIt->getOutputTensor(0);

                opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
                if (outputMemoryLocation.isForced())
                    opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
        }
    }
}