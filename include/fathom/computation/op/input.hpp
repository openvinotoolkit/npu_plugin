#ifndef INPUT_HPP_
#define INPUT_HPP_

#include "include/fathom/computation/op/source_op.hpp"

namespace mv
{

    class Input : public SourceOp
    {

    public:

        Input(Shape outputShape, DType dType, Order order, const string &name) :
        ComputationOp("input", name),
        SourceOp("input", name)
        {

            addAttr("shape", AttrType::ShapeType, outputShape);
            addAttr("dType", AttrType::DTypeType, dType);
            addAttr("order", AttrType::OrderType, order);
            addAttr("executable", AttrType::BoolType, false);

        }

        Tensor getOutputDef()
        {
            auto outputShape = getAttr("shape").getContent<Shape>();
            auto dType = getAttr("dType").getContent<DType>();
            auto order = getAttr("order").getContent<Order>();
            return Tensor(getOutputName(), outputShape, dType, order);
        }

    };

}

#endif // INPUT_HPP_