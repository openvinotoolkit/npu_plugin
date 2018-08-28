#ifndef INPUT_HPP_
#define INPUT_HPP_

#include "include/mcm/computation/op/source_op.hpp"

namespace mv
{

    namespace op
    {

        class Input : public SourceOp
        {

        public:

            Input(Shape outputShape, DType dType, Order order, const std::string &name);
            Input(mv::json::Value &obj);

            bool setOutputTensor(Data::TensorIterator &tensor, std::size_t idx);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // INPUT_HPP_
