#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <functional>
#include <string>

void removeSimpleOp(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType);
void removeShapeRelevant(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType);
void removeInterp(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType);
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

    mv::OpModel om(model);
    std::unordered_map<std::string, std::function<void(mv::Data::OpListIterator &, mv::ComputationModel& , std::string &)>> remTaskMap =
                                    {{"Identity", removeSimpleOp},
                                    {"Dropout", removeSimpleOp},
                                    {"Interp", removeInterp},
                                    {"Reshape", removeShapeRelevant},
                                    {"Permute", removeShapeRelevant},
                                    {"Slice", removeShapeRelevant}};

    std::vector<std::string> rem_types = {"Identity", "Dropout", "Interp", "Reshape", "Permute", "Slice"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfType = om.getOpsOfTypes(rem_types);
    for (auto type = rem_types.begin(); type != rem_types.end(); type++)
    {
        auto remFunctor = (remTaskMap.at(*type));
        for (auto opIt = operationsOfType[*type].begin(); opIt != operationsOfType[*type].end();++opIt)
            remFunctor(*opIt, model, *type);
    }
}

void removeSimpleOp(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{ //removeIdentityOps, removeDropOut, 
    UNUSED(opType);
    mv::OpModel om(model);
    auto sourceTensor = opIt->getInputTensor(0);
    auto parentOpIt = om.getSourceOp(sourceTensor);
    linkNewOperationsRemove(parentOpIt, om.tensorEnd(), om, opIt);
}

void removeShapeRelevant(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{ //removeInterpNoOpFcn, removeReshapeNoOpFcn, removePermuteNoOpFcn, removeSliceNoOpFcn
    mv::OpModel om(model);
    auto inputShape = opIt->getInputTensor(0)->getShape();
    auto outputShape = opIt->getOutputTensor(0)->getShape();
    if ((inputShape == outputShape) && // Sizes same, and either its not a slice, or additional conditions hold.
        ((opType != "Slice")||
         ((inputShape == opIt->get<mv::Shape>("size")) &&
         (opIt->get<mv::Shape>("begin").totalSize() == 0))))
    {
        auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto sourceTensor = parentOpIt->getOutputTensor(0);
        opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
        if (outputMemoryLocation.isForced())
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
    }
}

void removeInterp(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    auto inputShape = opIt->getInputTensor(0)->getShape();
    auto outputShape = opIt->getOutputTensor(0)->getShape();
    if (inputShape == outputShape)
    {
        auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
        auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto outQuantParams  = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");

        auto sourceTensor = parentOpIt->getOutputTensor(0);
        auto inQuantParams = sourceTensor->get<mv::QuantizationParams>("quantParams");

        if (isEqual(inQuantParams, outQuantParams) || outQuantParams.isNeutral())
        {
           opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
           if (outputMemoryLocation.isForced())
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
        }
    }
}
