#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

void removeIdentityOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void removeDropOut(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void removeInterpNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void removeReshapeNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void linkNewOperationsRemove(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                                    mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(RemoveOps)
        .setFunc(removeOpsFcn)
        .setDescription(
            "Removes Operations that do not need to be executed"
        );
    }

}

void removeOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                       mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    pass.log(mv::Logger::MessageType::Debug, "Removal passes are starting");
    removeIdentityOps(pass, model);
    removeDropOut(pass, model);
    removeInterpNoOpFcn(pass, model);
    removeReshapeNoOpFcn(pass, model);
}

mv::Data::OpListIterator linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt,
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

void removeIdentityOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto identityOps = om.getOps("Identity");

    for (auto& opIt : identityOps)
    {
        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        linkNewOperationsRemove(parentOpIt, om.tensorEnd(), om, opIt);
    }
}

void removeDropOut(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto dropoutOps = om.getOps("Dropout");
    for (auto& opIt : dropoutOps)
    {
        auto sourceTensor = opIt->getInputTensor(0);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        linkNewOperationsRemove(parentOpIt, om.tensorEnd(), om, opIt);
    }
}

void removeInterpNoOpFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto interpOps = om.getOps("Interp");
    for (auto& opIt : interpOps)
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

void removeReshapeNoOpFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto reshapeOps = om.getOps("Reshape");
    for (auto& opIt : reshapeOps)
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
