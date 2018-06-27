#ifndef COMPOSITIONAL_MODEL_HPP_
#define COMPOSITIONAL_MODEL_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"

namespace mv
{

    class CompositionalModel
    {
        
    public:

        virtual ~CompositionalModel() = 0;
        virtual Data::TensorIterator input(const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        virtual Data::TensorIterator output(Data::TensorIterator input, const string& name = "") = 0;
        virtual Data::TensorIterator constant(const dynamic_vector<float_type>& data, const Shape& shape, DType dType, Order order, const string& name = "") = 0;
        // padding [pad_left, pad_right, pad_top, pad_bottom]
        virtual Data::TensorIterator conv2D(Data::TensorIterator input, Data::TensorIterator filters, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") = 0;
        virtual Data::TensorIterator matMul(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") = 0;
        virtual Data::TensorIterator maxpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") = 0;
        virtual Data::TensorIterator avgpool2D(Data::TensorIterator input, UnsignedVector2D kernelSize, UnsignedVector2D stride, UnsignedVector4D padding, const string& name = "") = 0;
        virtual Data::TensorIterator concat(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") = 0;
        virtual Data::TensorIterator batchNorm(Data::TensorIterator input, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, float_type varianceEps, const string& name = "") = 0;
        virtual Data::TensorIterator scale(Data::TensorIterator input, Data::TensorIterator scale, const string& name = "") = 0;
        virtual Data::TensorIterator relu(Data::TensorIterator input, const string& name = "") = 0;
        virtual Data::TensorIterator softmax(Data::TensorIterator input, const string& name = "") = 0;
        virtual Data::TensorIterator add(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") = 0;
        virtual Data::TensorIterator subtract(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") = 0;
        virtual Data::TensorIterator multiply(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") = 0;
        virtual Data::TensorIterator divide(Data::TensorIterator input0, Data::TensorIterator input1, const string& name = "") = 0;
        virtual Data::TensorIterator reshape(Data::TensorIterator input, const Shape& shape, const string& name = "") = 0;
        virtual Data::TensorIterator bias(Data::TensorIterator input, Data::TensorIterator biases, const string& name = "") = 0;
        virtual Data::TensorIterator fullyConnected(Data::TensorIterator input, Data::TensorIterator weights, const string& name = "") = 0;

        virtual Data::OpListIterator getSourceOp(Data::TensorIterator tensor) = 0;
        virtual bool addAttr(Data::OpListIterator op, const string& name, const Attribute& attr) = 0;
        virtual bool isValid() const = 0;
        virtual bool isValid(const Data::TensorIterator &it) const = 0;
        virtual bool isValid(const Data::OpListIterator& it) const = 0;

    };

}

#endif //COMPOSITIONAL_MODEL_HPP_