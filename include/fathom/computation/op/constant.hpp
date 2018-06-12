#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_

#include "include/fathom/computation/op/source_op.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"

namespace mv
{

    class Constant : public SourceOp
    {

        allocator::owner_ptr<dynamic_vector<float_type>> data_;

    public:

        Constant(const dynamic_vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name) :
        ComputationOp(OpType::Constant, name),
        SourceOp(OpType::Constant, 1, name),
        data_(allocator_.make_owner<dynamic_vector<float_type>>(data))
        {
            addAttr("shape", AttrType::ShapeType, shape);
            addAttr("dType", AttrType::DTypeType, dType);
            addAttr("order", AttrType::OrderType, order);
            addAttr("executable", AttrType::BoolType, false);
        }

        Tensor getOutputDef(byte_type idx)
        {
            
            if (idx > 0)
                return Tensor();

            auto shape = getAttr("shape").getContent<Shape>();
            auto dType = getAttr("dType").getContent<DType>();
            auto order = getAttr("order").getContent<Order>();
            return Tensor(name_ + ":0", shape, dType, order, data_);
        }

    };

}

#endif // CONSTANT_HPP_