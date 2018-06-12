#ifndef COMPOSITIONAL_MODEL_HPP_
#define COMPOSITIONAL_MODEL_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/model/attribute.hpp"

namespace mv
{

    class CompositionalModel
    {

    public:

        virtual ~CompositionalModel() = 0;
        virtual DataContext::OpListIterator input(const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        virtual DataContext::OpListIterator output(DataContext::TensorIterator input, const string& name = "") = 0;
        virtual DataContext::OpListIterator constant(float_type *data, size_type size, const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        virtual DataContext::OpListIterator constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        // padding [pad_left, pad_right, pad_top, pad_bottom]
        virtual DataContext::OpListIterator conv2D(DataContext::TensorIterator input, DataContext::TensorIterator filters,
        UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") = 0;
        virtual DataContext::OpListIterator fullyConnected(DataContext::TensorIterator input, DataContext::TensorIterator weights, const string& name) = 0;
        virtual DataContext::OpListIterator maxpool2D(DataContext::TensorIterator input, UnsignedVector2D kernelSize,
        UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") = 0;
        virtual DataContext::OpListIterator concat(DataContext::TensorIterator input0, DataContext::TensorIterator input1, const string& name = "") = 0;
        virtual bool addAttr(DataContext::OpListIterator& op, const string& name, const Attribute& attr) = 0;
        virtual bool isValid() const = 0;

    };

}

#endif //COMPOSITIONAL_MODEL_HPP_