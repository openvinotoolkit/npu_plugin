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

            Input(Shape outputShape, DType dType, Order order, const string &name);
            Input(mv::json::Value &obj);

            bool setOutputTensor(Data::TensorIterator &tensor, byte_type idx);
            Tensor getOutputDef(byte_type idx);

        };

    }

}

#endif // INPUT_HPP_
