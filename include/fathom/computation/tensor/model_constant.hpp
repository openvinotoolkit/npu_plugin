#ifndef MODEL_CONSTANT_TENSOR_HPP_
#define MODEL_CONSTANT_TENSOR_HPP_

#include "include/fathom/computation/tensor/model_tensor.hpp"
#include "include/fathom/computation/tensor/constant.hpp"

namespace mv
{

    class ConstantModelTensor : public ModelTensor
    {

        vector<float_type> data_;

    public:

        ConstantModelTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order, vector<float_type> data) : 
        ModelTensor(logger, "CT_" + name, shape, dType, order)
        {
            assert(data.size() == shape.totalSize() && "Mismatch between declared size of tensor and size of its initialization data");
            data_ = data;
        }

        ConstantModelTensor(const Logger &logger, const string &name, const ConstantTensor &tensor) :
        ModelTensor(logger, "CT_" + name, tensor.getShape(), tensor.getDType(), tensor.getOrder()) 
        {
            auto data = tensor.getData();
            assert(data.size() == shape_.totalSize() && "Mismatch between declared size of tensor and size of its initialization data");
            data_ = data;
        }

        Shape getShape() const
        {
            return this->shape_;
        }

        string toString() const
        {
            return string("Const tensor " + name_ + " " + shape_.toString());
        }


    };

}

#endif // CONSTANT_TENSOR_HPP_