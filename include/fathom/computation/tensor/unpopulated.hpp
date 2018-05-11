#ifndef UNPOPULATED_TENSOR_HPP_
#define UNPOPULATED_TENSOR_HPP_

#include "include/fathom/computation/tensor/model_tensor.hpp"

namespace mv
{

    class UnpopulatedTensor : public ModelTensor
    {

    public:

        UnpopulatedTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order) : 
        ModelTensor(logger, "vt_" + name, shape, dType, order)
        {

        }

        UnpopulatedTensor(const UnpopulatedTensor &other) :
        ModelTensor(other)
        {

        }

        string toString() const
        {
            return string("var tensor " + name_ + " " + shape_.toString());
        }

    };

}

#endif // UNPOPULATED_TENSOR_HPP_