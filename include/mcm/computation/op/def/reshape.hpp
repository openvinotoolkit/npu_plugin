#ifndef RESHAPE_HPP_
#define RESHAPE_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{

    namespace Op
    {

        class Reshape : public SourceOp, public SinkOp
        {

        public:

            Reshape(Shape outputShape, const string& name) :
            ComputationOp(OpType::Reshape, name),
            SourceOp(OpType::Reshape, 1, name),
            SinkOp(OpType::Reshape, 1, name)
            {
                addAttr("shape", AttrType::ShapeType, outputShape);
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
                auto outputShape = getAttr("shape").getContent<Shape>();

                if (inputShape.totalSize() != outputShape.totalSize())
                {
                    logger_.log(Logger::MessageType::MessageError, "Unable to define output tensor for '" + name_ + 
                        "' because conversion of input shape " + inputShape.toString() + " and requested shape " + outputShape.toString() +
                        " is impossible");
                    return Tensor();
                }

                return Tensor(name_ + ":0", outputShape, input->getDType(), input->getOrder());

            }

        };

    }

}

#endif // RESHAPE_HPP_