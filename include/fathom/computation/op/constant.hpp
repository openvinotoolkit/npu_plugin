#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_

#include "include/fathom/computation/op/source_op.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"

namespace mv
{

    class Constant : public SourceOp
    {

        allocator::owner_ptr<vector<float_type>> data_;

    public:

        Constant(const vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name) :
        ComputationOp("const", name),
        SourceOp("const", name),
        data_(allocator_.make_owner<vector<float_type>>(data))
        {
            addAttr("shape", AttrType::ShapeType, shape);
            addAttr("dType", AttrType::DTypeType, dType);
            addAttr("order", AttrType::OrderType, order);
            addAttr("executable", AttrType::BoolType, false);
        }

        Tensor getOutputDef()
        {
            auto shape = getAttr("shape").getContent<Shape>();
            auto dType = getAttr("dType").getContent<DType>();
            auto order = getAttr("order").getContent<Order>();
            return Tensor(getOutputName(), shape, dType, order, data_);
        }

    };

}

#endif // CONSTANT_HPP_