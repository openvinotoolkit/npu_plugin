#ifndef MODEL_POPULATED_TENSOR_HPP_
#define MODEL_POPULATED_TENSOR_HPP_

#include "include/fathom/computation/tensor/model_tensor.hpp"
#include "include/fathom/computation/tensor/constant.hpp"

namespace mv
{

    class PopulatedModelTensor : public ModelTensor
    {

        ConstantTensor &data_;

    public:
    
        PopulatedModelTensor(const Logger &logger, const string &name, ConstantTensor &data) :
        ModelTensor(logger, "ct_" + name, data),
        data_(data)
        {

        }

        PopulatedModelTensor(const PopulatedModelTensor &other) :
        ModelTensor(other),
        data_(other.data_)
        {

        }

        Shape getShape() const
        {
            return this->shape_;
        }

        string toString() const
        {
            return string("const tensor " + name_ + " " + shape_.toString());
        }

    };

}

#endif // MODEL_POPULATED_TENSOR_HPP_