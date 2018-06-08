#ifndef COMPOSITIONAL_MODEL_HPP_
#define COMPOSITIONAL_MODEL_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/model/attribute.hpp"

namespace mv
{

    class CompositionalModel
    {

    public:

        virtual DataContext::OpListIterator input(const Shape &shape, DType dType, Order order, const string &name = "") = 0;
        virtual DataContext::OpListIterator output(DataContext::OpListIterator &input, const string &name = "") = 0;
        virtual DataContext::OpListIterator conv(DataContext::OpListIterator &input, DataContext::OpListIterator &weights, byte_type strideX, 
        byte_type strideY, byte_type padX, byte_type padY, const string &name = "") = 0;
        virtual DataContext::OpListIterator maxpool(DataContext::OpListIterator &input, const Shape &kernelShape, byte_type strideX, 
        byte_type strideY, byte_type padX, byte_type padY, const string &name = "") = 0;
        virtual DataContext::OpListIterator concat(DataContext::OpListIterator &input0, DataContext::OpListIterator &input1, const string &name = "") = 0;
        virtual DataContext::OpListIterator constant(float_type *data, size_type size, const Shape &shape, DType dType, Order order, const string &name = "") = 0;
        virtual DataContext::OpListIterator constant(const vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name = "") = 0;
        virtual bool addAttr(DataContext::OpListIterator &op, const string &name, const Attribute &attr) = 0;
        virtual bool isValid() const = 0;

    };

}

#endif //COMPOSITIONAL_MODEL_HPP_