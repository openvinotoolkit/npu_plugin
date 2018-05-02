#ifndef CONSTANT_TENSOR_HPP_
#define CONSTANT_TENSOR_HPP_

#include "include/fathom/computation/tensor/tensor.hpp"

namespace mv
{

    class ConstantTensor : public Tensor
    {

        vector<float_type> data_;

    public:

        ConstantTensor(const Shape &shape, DType dType, Order order, vector<float_type> data) : 
        Tensor(shape, dType, order)
        {
            assert(data.size() == shape.totalSize() && "Mismatch between declared size of tensor and size of its initialization data");
            data_ = data;
        }

        vector<float_type> getData() const
        {
            return data_;
        }

    };

}

#endif // CONSTANT_TENSOR_HPP_