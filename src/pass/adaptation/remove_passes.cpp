#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"

void removeIdentityOps(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void removeDropOut(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
void removeInterpNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void removeReshapeNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void removePermuteNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
void removeSliceNoOpFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
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
    removeSliceNoOpFcn(pass, model);
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
            auto outQuantParams  = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");

            auto sourceTensor = parentOpIt->getOutputTensor(0);
            auto inQuantParams = sourceTensor->get<mv::QuantizationParams>("quantParams");

            if (isEqual(inQuantParams, outQuantParams) || outQuantParams.isNeutral())
            {
               opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
               if (outputMemoryLocation.isForced())
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
            else
            {
                pass.log(mv::Logger::MessageType::Debug, "Replacing with DW requanitze");

                  //FIND THE APPROPRIATE FLOW
                mv::Data::TensorIterator weights;
                std::vector<int64_t> zp = { 0 };
                std::vector<double> min = { 1 };
                std::vector<double> max = { 1 };
                std::vector<double> scale = { 1 };
                mv::QuantizationParams weightsQuantParams(zp, scale, min, max);
                int64_t weightsValue = 1;
                std::vector<int64_t> weightsData(sourceTensor->getShape()[mv::IO_CHANNEL_DIMENSION], weightsValue);
                weights = om.constantInt(weightsData,
                        {1, 1, sourceTensor->getShape()[mv::IO_CHANNEL_DIMENSION], 1},
                        mv::DType("UInt8"),
                        mv::Order(mv::Order::getRowMajorID(4)),
                        weightsQuantParams);
                auto reQuantizeDepthwise = om.depthwiseConv(sourceTensor, weights, {1,1}, {0, 0, 0, 0},
                        1, mv::DType("UInt8"), {outQuantParams.getZeroPoint(),outQuantParams.getScale(),{},{}}, opIt->getName() + "_DepthwiseRequantize");
                auto reQuantizeDepthwiseOp = om.getSourceOp(reQuantizeDepthwise);
                auto weightsOp = om.getSourceOp(weights);
                reQuantizeDepthwiseOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                weightsOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                linkNewOperationsReplacement(parentOpIt, reQuantizeDepthwise, om, opIt);
                reQuantizeDepthwise->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
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

void removeSliceNoOpFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    auto sliceOps = om.getOps("Slice");
    for (auto& opIt : sliceOps)
    {
        auto inputShape = opIt->getInputTensor(0)->getShape();
        auto outputShape = opIt->getOutputTensor(0)->getShape();
        if ((inputShape == outputShape) && 
            (inputShape == opIt->get<mv::Shape>("size")) &&
            (opIt->get<mv::Shape>("begin").totalSize() == 0))
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
