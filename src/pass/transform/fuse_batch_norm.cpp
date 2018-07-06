#include "include/mcm/pass/transform/fuse_batch_norm.hpp"

mv::pass::FuseBatchNorm::FuseBatchNorm() :
TransformPass("FuseBatchNormPass")
{

}

bool mv::pass::FuseBatchNorm::run_(ComputationModel &model)
{
    
    OpModel om(model);

    bool result = true;

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::BatchNorm)
        {
            
            auto batchNormName = opIt->getName();
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            ControlModel cm(om);
            auto nextOp = cm.switchContext(opIt).leftmostChild();
            
            auto bnMean = *opIt->getInputTensor(1);
            auto bnVar = *opIt->getInputTensor(2);
            auto bnOffset = *opIt->getInputTensor(3);
            auto bnScale = *opIt->getInputTensor(4);
            float bnEps = opIt->getAttr("varianceEps").getContent<float>();

            auto scaleParam = math::divide(bnScale, math::sqrt(math::add(bnVar, bnEps)));
            auto offsetParam = math::subtract(bnOffset, math::multiply(bnMean, scaleParam));
            auto offset = om.constant(offsetParam.getData(), offsetParam.getShape(), offsetParam.getDType(), offsetParam.getOrder());

            om.disableDefaultControlFlow();

            Data::TensorIterator sourceTensor;

            if (bnMean.getShape().ndims() == 1)
            {
                if (parentOpIt->getOpType() == OpType::Conv2D)
                {
                    parentOpIt->getInputTensor(1)->mulitply(scaleParam);
                    sourceTensor = parentOpIt->getOutputTensor(0);
                }
                else
                {
                    auto scale = om.constant(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
                    sourceTensor = om.scale(opIt->getInputTensor(0), scale);
                    cm.defineFlow(parentOpIt, om.getSourceOp(sourceTensor));
                    parentOpIt = om.getSourceOp(sourceTensor);
                }
            }
            else
            {
                auto scale = om.constant(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
                sourceTensor = om.multiply(opIt->getInputTensor(0), scale);
                cm.defineFlow(parentOpIt, om.getSourceOp(sourceTensor));
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

            cm.defineFlow(parentOpIt, om.getSourceOp(sourceTensor));
            cm.defineFlow(om.getSourceOp(sourceTensor), om.switchContext(nextOp));

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
            om.enableDefaultControlFlow(cm.getLast());
            opIt = parentOpIt;
        }

    }

    return result;

}