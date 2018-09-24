#ifndef CONSTANT_HPP_
#define CONSTANT_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/tensor/tensor.hpp"

namespace mv
{

    namespace op
    {

        class Constant : public SourceOp
        {

            std::vector<double> data_;

        public:

            Constant(const std::vector<double> &data, const Shape &shape, DType dType, Order order, const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}

#endif // CONSTANT_HPP_
