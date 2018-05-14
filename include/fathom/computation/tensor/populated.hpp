#ifndef POPULATED_TENSOR_HPP_
#define POPULATED_TENSOR_HPP_

#include "include/fathom/computation/tensor/model_tensor.hpp"
#include "include/fathom/computation/tensor/constant.hpp"

namespace mv
{

    class PopulatedTensor : public ModelTensor
    {

        ConstantTensor &data_;

    public:
    
        PopulatedTensor(const Logger &logger, const string &name, ConstantTensor &data) :
        ModelTensor(logger, "ct_" + name, data),
        data_(data)
        {
            //logger_.log(Logger::MessageType::MessageInfo, "Defined populated tensor '" + toString());
        }

        PopulatedTensor(const PopulatedTensor &other) :
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
            return "populated tensor '" + name_ + "' " + ComputationElement::toString();
        }

    };

}

#endif // POPULATED_TENSOR_HPP_