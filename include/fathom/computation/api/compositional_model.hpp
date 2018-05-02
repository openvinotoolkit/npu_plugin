#ifndef COMPOSITIONAL_MODEL_HPP_
#define COMPOSITIONAL_MODEL_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/constant.hpp"


namespace mv
{

    class CompositionalModel
    {

    public:

        virtual const OpListIterator input(const Shape &shape, Tensor::DType dType, Tensor::Order order, const string &name = "") = 0;
        virtual const OpListIterator output(OpListIterator &predecessor, const string &name = "") = 0;
        virtual OpListIterator convolutional(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, byte_type strideY, const string &name = "") = 0;
        /*virtual bool attr(OpListIterator &op, const string &name, const float_type &val) = 0;
        virtual bool attr(OpListIterator &op, const string &name, const int_type &val) = 0;
        virtual bool attr(OpListIterator &op, const string &name, const ConstantTensor &val) = 0;*/

    };

}

#endif //COMPOSITIONAL_MODEL_HPP_