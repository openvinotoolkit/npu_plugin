#include "include/mcm/pass/transform/fuse_scale.hpp"

mv::pass::FuseScale::FuseScale() :
TransformPass("FuseScalePass")
{

}

bool mv::pass::FuseScale::run_(ComputationModel &model)
{
    
    OpModel om(model);
    bool result = true;

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {   

        if (opIt->getOpType() == OpType::Scale)
        {
            
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
            
            if (parentOpIt->getOpType() == OpType::Conv2D)
            {

                auto scale = *opIt->getInputTensor(1);
                parentOpIt->getInputTensor(1)->mulitply(scale);

                if (parentOpIt->hasAttr("bias"))
                {
                    auto biasData = parentOpIt->getAttr("bias").getContent<dynamic_vector<float>>();
                    if (biasData.size() != scale.getData().size())
                        return false;

                    for (unsigned i = 0; i < biasData.size(); ++i)
                        biasData[i] *= scale.getData()[i];
                    
                    parentOpIt->getAttr("bias").setContent<dynamic_vector<float>>(biasData);

                }

                ControlModel cm(om);
                auto nextOp = cm.switchContext(opIt).leftmostChild();

                cm.defineFlow(parentOpIt, om.switchContext(nextOp));
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

    return result;

}