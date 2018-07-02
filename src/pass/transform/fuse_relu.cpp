#include "include/mcm/pass/transform/fuse_relu.hpp"

mv::pass::FuseReLU::FuseReLU() :
TransformPass("FuseReLUPass")
{

}

bool mv::pass::FuseReLU::run_(ComputationModel &model)
{
    
    OpModel om(model);
    bool result = true;

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::ReLU)
        {

            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            Attribute reluAttr(AttrType::OpTypeType, OpType::ReLU);
            om.addAttr(parentOpIt, "postOpType", reluAttr);

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

    return result;

}