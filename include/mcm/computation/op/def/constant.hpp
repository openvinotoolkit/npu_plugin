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

            std::vector<double> data_;

        public:

            Constant(const std::vector<double> &data, const Shape &shape, DType dType, Order order, const std::string &name);
            Constant(mv::json::Value &obj);
            Tensor getOutputDef(std::size_t idx);
            mv::json::Value toJsonValue() const;
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // CONSTANT_HPP_
