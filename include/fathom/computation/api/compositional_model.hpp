#ifndef COMPOSITIONAL_MODEL_HPP_
#define COMPOSITIONAL_MODEL_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/tensor/constant.hpp"
#include "include/fathom/computation/model/attribute.hpp"

namespace mv
{

    class CompositionalModel
    {

    public:

        virtual const OpListIterator input(const Shape &shape, DType dType, Order order, const string &name = "") = 0;
        virtual const OpListIterator output(OpListIterator &predecessor, const string &name = "") = 0;
        virtual OpListIterator conv2D(OpListIterator &predecessor, const ConstantTensor &weights, byte_type strideX, 
        byte_type strideY, byte_type padX, byte_type padY, const string &name = "") = 0;
        virtual bool addAttr(OpListIterator &op, const string &name, const Attribute &attr) = 0;
        virtual bool isValid() const = 0;

    };

}

#endif //COMPOSITIONAL_MODEL_HPP_