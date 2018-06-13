#ifndef SCALE_HPP_
#define SCALE_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{

    namespace Op
    {

        class Scale : public SourceOp, public SinkOp
        {

        public:

            Scale(const string &name) :
            ComputationOp(OpType::Scale, name),
            SourceOp(OpType::Scale, 1, name),
            SinkOp(OpType::Scale, 2, name)
            {
                addAttr("executable", AttrType::BoolType, true);
            }

            Tensor getOutputDef(byte_type idx)
            {
                
                if (idx > 0)
                    return Tensor();

                if (!validOutputDef_())
                    return Tensor();

                auto input = getInput(0);
                auto inputShape = input->getShape(); 

                auto scale = getInput(1);
                auto scaleShape = scale->getShape();

                if (inputShape != scaleShape)
                {

                    if (scaleShape.ndims() != 1 || scaleShape[0] != inputShape[-1])
                    {
                        logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                            "' because of incorrect shape of scale (" + scaleShape.toString() + ") - it needs to be either"
                            " equal to shape of the input (" + inputShape.toString() + ") or to be one dimensional tensors of dimension " +
                            Printable::toString(inputShape[-1]));
                        return Tensor();
                    }

                }

                return Tensor(name_ + ":0", inputShape, input->getDType(), input->getOrder());
                
            }

        };

    }

}

#endif // BATCH_NORM_HPP_