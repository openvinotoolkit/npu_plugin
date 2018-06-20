#ifndef FUSE_BATCH_NORM_HPP_
#define FUSE_BATCH_NORM_HPP_

#include "include/mcm/pass/transform_pass.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"

namespace mv
{

    namespace pass
    {

        class FuseBatchNorm : public TransformPass
        {

            bool run_(ComputationModel &model)
            {
                
                OpModel om(model);
                bool result = true;

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
                        auto offset = om.constant(offsetParam.getData(), offsetParam.getShape(), offsetParam.getDType(), offsetParam.getOrder());

                        Data::TensorIterator sourceTensor;

                        if (parentOpIt->getOpType() == OpType::Conv2D)
                        {
                            parentOpIt->getInputTensor(1)->mulitply(scaleParam);
                            sourceTensor = opIt->getInputTensor(0);
                        }
                        else
                        {
                            auto scale = om.constant(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
                            sourceTensor = om.multiply(opIt->getInputTensor(0), scale);
                        }

                        //parentOpIt->getInputTensor(1)->mulitply(math::divide(scale, math::sqrt(var)));
                        //auto biasesTensor = math::subtract(offset, mv::math::divide(scale, math::sqrt(var)));
                        //auto biases = om.constant(biasesTensor.getData(), biasesTensor.getShape(), 
                        //    opIt->getInputTensor(0)->getDType(), opIt->getInputTensor(0)->getOrder());

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

                    }

                }

                return result;

            }

        public:

            FuseBatchNorm() :
            TransformPass("FuseBatchNormPass")
            {

            }

        };

    }

}

#endif // DEPLOY_PASS_HPP_