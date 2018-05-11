#ifndef MODEL_UNPOPULATED_TENSOR_HPP_
#define MODEL_UNPOPULATED_TENSOR_HPP_

#include "include/fathom/computation/tensor/model_tensor.hpp"

namespace mv
{

    class UnpopulatedModelTensor : public ModelTensor
    {

    public:

        UnpopulatedModelTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order) : 
        ModelTensor(logger, "vt_" + name, shape, dType, order)
        {

        }

        UnpopulatedModelTensor(const UnpopulatedModelTensor &other) :
        ModelTensor(other)
        {

        }

        string toString() const
        {
            return string("var tensor " + name_ + " " + shape_.toString());
        }

    };

}

#endif // MODEL_UNPOPULATED_TENSOR_HPP_