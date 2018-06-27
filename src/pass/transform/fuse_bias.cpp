#include "include/mcm/pass/transform/fuse_bias.hpp"

mv::pass::FuseBias::FuseBias() :
TransformPass("FuseBiasPass")
{

}

bool mv::pass::FuseBias::run_(ComputationModel &model)
{
    
    OpModel om(model);
    bool result = true;

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::Bias)
        {
            
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            
            if (parentOpIt->getOpType() == OpType::Conv2D)
            {

                auto bias = *opIt->getInputTensor(1);
                Attribute biasAttr(AttrType::FloatVecType, bias.getData());
                om.addAttr(parentOpIt, "bias", biasAttr);

                ControlModel cm(om);
                cm.defineFlow(parentOpIt,  opIt.leftmostChild());
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
                om.enableDefaultControlFlow(om.getSourceOp(sourceTensor));

            }

        }

    }

    return result;

}