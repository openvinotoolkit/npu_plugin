#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"

static void fuseBatchNormFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void fuseBiasFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void fuseReluFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);
static void fuseScaleFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(FuseBatchNorm)
        .setFunc(fuseBatchNormFcn)
        .setGenre(PassGenre::Adaptation)
        .setDescription(
            "Replaces a batchnorm op with eltwise ops or their 1d equivalents. "
            "Following cases are handled:\n"
            " - batchnorm parameters tensors are n-dimensional - the replacement is a chain of multiply and add (eltwise)\n"
            " - batchnorm parameters tensors are 1-dimensional and parent op is not of type conv2d - the replacement is chain of scale and bias\n"
            " - batchnorm parameters tensors are 1-dimensional and parent op is of type conv2d - the replacement is a bias and weights of conv2d are modified]\n"
        );

        MV_REGISTER_PASS(FuseBias)
        .setFunc(fuseBiasFcn)
        .setGenre(PassGenre::Adaptation)
        .setDescription(
            "Fuses a bias op to the parent op if this op is of type conv2d. "
            "Bias op is removed from the model, the biases tensor "
            " (second input of bias op) is appended to parent op as an attribute "
            " of type float vector of name 'bias'."
        );

        MV_REGISTER_PASS(FuseRelu)
        .setFunc(fuseReluFcn)
        .setGenre(PassGenre::Adaptation)
        .setDescription(
            "Fuses a relu op to the parent op."
            "Relu op is removed from the model, and a new attribute of type OpType and value relu is defined for parent Op."
        );

        MV_REGISTER_PASS(FuseScale)
        .setFunc(fuseScaleFcn)
        .setGenre(PassGenre::Adaptation)
        .setDescription(
            "Fuses a scale op to the parent op is this op is of type conv2d. Scale op is removed from the model and conv2d weights are rescaled. "
            "If the conv2d has a 'bias' attribute of type float vector it is rescaled as well. In this case the length of scale op parameters tensor and 'bias' attribute "
            "has to match."
        );

    }

}

void fuseBiasFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::Bias)
        {
            
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            
            if (parentOpIt->getOpType() == OpType::Conv2D || parentOpIt->getOpType() == OpType::FullyConnected)
            {

                auto bias = *opIt->getInputTensor(1);

                if (parentOpIt->hasAttr("bias"))
                {
                    auto biasData = parentOpIt->getAttr("bias").getContent<dynamic_vector<float>>();
                    for (std::size_t i = 0; i < biasData.size(); ++i)
                        biasData[i] += bias.getData()[i];
                    parentOpIt->getAttr("bias").setContent<dynamic_vector<float>>(biasData);
                }
                else
                {
                    Attribute biasAttr(AttrType::FloatVecType, bias.getData());
                    om.addAttr(parentOpIt, "bias", biasAttr);
                }
                
                auto sourceTensor = parentOpIt->getOutputTensor(0);

                for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
                {
                    byte_type inputIdx = sinkFlow->getAttr("sinkInput").getContent<byte_type>();
                    sinkFlow.sink()->removeAttr("input" + Printable::toString(inputIdx));
                    om.defineFlow(sourceTensor, sinkFlow.sink(), inputIdx); 
                }

                while(opIt.parentsSize() > 1)
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

}

void fuseScaleFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    
    using namespace mv;
    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::Scale)
        {
            
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            
            if (parentOpIt->getOpType() == OpType::Conv2D)
            {

                auto scale = *opIt->getInputTensor(1);
                parentOpIt->getInputTensor(1)->multiply(scale);

                if (parentOpIt->hasAttr("bias"))
                {
                    auto biasData = parentOpIt->getAttr("bias").getContent<dynamic_vector<float>>();
                    if (biasData.size() != scale.getData().size())
                        throw pass::RutimeError("Mismatch between bias length and scale length");

                    for (unsigned i = 0; i < biasData.size(); ++i)
                        biasData[i] *= scale.getData()[i];
                    
                    parentOpIt->getAttr("bias").setContent<dynamic_vector<float>>(biasData);

                }

                auto sourceTensor = parentOpIt->getOutputTensor(0);

                for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
                {
                    byte_type inputIdx = sinkFlow->getAttr("sinkInput").getContent<byte_type>();
                    sinkFlow.sink()->removeAttr("input" + Printable::toString(inputIdx));
                    om.defineFlow(sourceTensor, sinkFlow.sink(), inputIdx); 
                }

                while(opIt.parentsSize() > 1)
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

}

void fuseReluFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{
    
    using namespace mv;
    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::ReLU)
        {

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            Attribute reluAttr(AttrType::OpTypeType, OpType::ReLU);
            om.addAttr(parentOpIt, "postOpType", reluAttr);

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                byte_type inputIdx = sinkFlow->getAttr("sinkInput").getContent<byte_type>();
                sinkFlow.sink()->removeAttr("input" + Printable::toString(inputIdx));
                om.defineFlow(sourceTensor, sinkFlow.sink(), inputIdx); 
            }

            while(opIt.parentsSize() > 1)
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

void fuseBatchNormFcn(mv::ComputationModel& model, mv::TargetDescriptor&, mv::json::Object&, mv::json::Object&)
{

    using namespace mv;   
    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::BatchNorm)
        {
            
            auto batchNormName = opIt->getName();
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            
            auto bnMean = *opIt->getInputTensor(1);
            auto bnVar = *opIt->getInputTensor(2);
            auto bnOffset = *opIt->getInputTensor(3);
            auto bnScale = *opIt->getInputTensor(4);
            float bnEps = opIt->getAttr("varianceEps").getContent<float>();

            auto scaleParam = math::divide(bnScale, math::sqrt(math::add(bnVar, bnEps)));
            auto offsetParam = math::subtract(bnOffset, math::multiply(bnMean, scaleParam));
            auto offset = om.constant(offsetParam.getData(), offsetParam.getShape(), offsetParam.getDType(), offsetParam.getOrder(), batchNormName + "_offset");

            Data::TensorIterator sourceTensor;

            if (bnMean.getShape().ndims() == 1)
            {
                if (parentOpIt->getOpType() == OpType::Conv2D)
                {
                    parentOpIt->getInputTensor(1)->multiply(scaleParam);
                    sourceTensor = parentOpIt->getOutputTensor(0);
                }
                else
                {
                    auto scale = om.constant(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
                    sourceTensor = om.scale(opIt->getInputTensor(0), scale);
                    parentOpIt = om.getSourceOp(sourceTensor);
                }
            }
            else
            {
                auto scale = om.constant(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
                sourceTensor = om.multiply(opIt->getInputTensor(0), scale);
                parentOpIt = om.getSourceOp(sourceTensor);

            }

            if (offsetParam.getShape().ndims() == 1)
            {
                sourceTensor = om.bias(sourceTensor, offset); 
            }   
            else
            {
                sourceTensor = om.add(sourceTensor, offset);
            }

            for (Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                byte_type inputIdx = sinkFlow->getAttr("sinkInput").getContent<byte_type>();
                sinkFlow.sink()->removeAttr("input" + Printable::toString(inputIdx));
                om.defineFlow(sourceTensor, sinkFlow.sink(), inputIdx); 
            }

            while(opIt.parentsSize() > 1)
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