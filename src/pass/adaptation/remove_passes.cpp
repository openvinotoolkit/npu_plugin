#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"

void removeIdentityOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void removeDropOut(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void removeInterpNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void removeReshapeNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void removePermuteNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void replacePoolReshapePatternFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
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

void removePermuteNoOpFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto permuteOps = om.getOps("Permute");
    for (auto& opIt : permuteOps)
    {
        auto inputShape = opIt->getInputTensor(0)->getShape();
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        if ((inputShape == outputShape) || (inputShape.isFlat() && outputShape.isFlat()))
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
