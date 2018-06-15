#ifndef INPUT_HPP_
#define INPUT_HPP_

#include "include/mcm/computation/op/source_op.hpp"

namespace mv
{

    namespace Op
    {

        class Input : public SourceOp
        {

        public:

            Input(Shape outputShape, DType dType, Order order, const string &name) :
            ComputationOp(OpType::Input, name),
            SourceOp(OpType::Input, 1, name)
            {

                addAttr("shape", AttrType::ShapeType, outputShape);
                addAttr("dType", AttrType::DTypeType, dType);
                addAttr("order", AttrType::OrderType, order);
                addAttr("executable", AttrType::BoolType, false);

            }

            virtual bool setOutput(Data::TensorIterator &tensor, byte_type idx)
            {

                bool result = SourceOp::setOutput(tensor, idx);
                return result;

            }

            Tensor getOutputDef(byte_type idx)
            {

                if (idx > 0)
                    return Tensor();

                auto outputShape = getAttr("shape").getContent<Shape>();
                auto dType = getAttr("dType").getContent<DType>();
                auto order = getAttr("order").getContent<Order>();
                return Tensor(name_ + ":0", outputShape, dType, order);

            }

        };

    }

}

#endif // INPUT_HPP_