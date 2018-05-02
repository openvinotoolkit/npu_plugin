#ifndef MODEL_TENSOR_HPP_
#define MODEL_TENSOR_HPP_

#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/model/element.hpp"
#include "include/fathom/computation/logger/logger.hpp"
#include "include/fathom/computation/tensor/tensor.hpp"

namespace mv
{

    class ModelTensor : public Tensor, public ComputationElement
    {
    
    public:

        ModelTensor(const Logger &logger, const string &name, const Shape &shape, DType dType, Order order) : 
        Tensor(shape, dType, order),
        ComputationElement(logger, name)
        {

        }

        ModelTensor(const Logger &logger, const string &name, const Tensor &tensor) :
        Tensor(tensor),
        ComputationElement(logger, name)
        {

        }

    };

}

#endif // MODEL_TENSOR_HPP_