#ifndef MODEL_VARIABLE_TENSOR_HPP_
#define MODEL_VARIABLE_TENSOR_HPP_

#include "include/fathom/computation/tensor/model_tensor.hpp"

namespace mv
{

    class VariableTensor : public ModelTensor
    {

    public:

        VariableTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order) : 
        ModelTensor(logger, "vt_" + name, shape, dType, order)
        {

        }

        string toString() const
        {
            return string("var tensor " + name_ + " " + shape_.toString());
        }

    };

}

#endif // MODEL_VARIABLE_TENSOR_HPP_