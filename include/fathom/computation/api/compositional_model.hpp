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
        virtual DataContext::TensorIterator input(const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        virtual DataContext::TensorIterator output(DataContext::TensorIterator input, const string& name = "") = 0;
        virtual DataContext::TensorIterator constant(float_type *data, size_type size, const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        virtual DataContext::TensorIterator constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        // padding [pad_left, pad_right, pad_top, pad_bottom]
        virtual DataContext::TensorIterator conv2D(DataContext::TensorIterator input, DataContext::TensorIterator filters, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") = 0;
        virtual DataContext::TensorIterator fullyConnected(DataContext::TensorIterator input, DataContext::TensorIterator weights, const string& name) = 0;
        virtual DataContext::TensorIterator maxpool2D(DataContext::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") = 0;
        virtual DataContext::TensorIterator concat(DataContext::TensorIterator input0, DataContext::TensorIterator input1, const string& name = "") = 0;
        virtual DataContext::TensorIterator batchNorm(DataContext::TensorIterator input, DataContext::TensorIterator mean, DataContext::TensorIterator variance, DataContext::TensorIterator offset, DataContext::TensorIterator scale, float_type varianceEps, const string& name = "") = 0;
        virtual DataContext::TensorIterator scale(DataContext::TensorIterator input, DataContext::TensorIterator scale, const string& name = "") = 0;
        virtual DataContext::TensorIterator relu(DataContext::TensorIterator input, const string& name = "") = 0;

        virtual DataContext::OpListIterator getSourceOp(DataContext::TensorIterator tensor) = 0;
        virtual bool addAttr(DataContext::OpListIterator op, const string& name, const Attribute& attr) = 0;
        virtual bool isValid() const = 0;

    };

}

#endif //COMPOSITIONAL_MODEL_HPP_