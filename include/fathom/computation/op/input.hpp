#ifndef INPUT_HPP_
#define INPUT_HPP_

#include "include/fathom/computation/op/source_op.hpp"

namespace mv
{

    class Input : public SourceOp
    {

    public:

        Input(const Logger &logger, Shape outputShape, DType dType, Order order, const string &name) :
        ComputationOp(logger, "input", name),
        SourceOp(logger, "input", name)
        {

            addAttr("shape", AttrType::ShapeType, outputShape);
            addAttr("dType", AttrType::DTypeType, dType);
            addAttr("order", AttrType::OrderType, order);
            addAttr("executable", AttrType::BoolType, false);

        }

        UnpopulatedTensor getOutputDef()
        {
            auto outputShape = getAttr("shape").getContent<Shape>();
            auto dType = getAttr("dType").getContent<DType>();
            auto order = getAttr("order").getContent<Order>();
            return UnpopulatedTensor(logger_, getOutputName(), outputShape, dType, order);
        }

    };

}

#endif // INPUT_HPP_