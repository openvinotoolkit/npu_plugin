#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"

namespace mv
{

    namespace op
    {

        class Constant : public SourceOp
        {

            dynamic_vector<float_type> data_;

        public:

            Constant(const dynamic_vector<float_type> &data, const Shape &shape, DType dType, Order order, const string &name);
            Tensor getOutputDef(byte_type idx);

        };

    }

}

#endif // CONSTANT_HPP_