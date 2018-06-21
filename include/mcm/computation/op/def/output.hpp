#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{

    namespace op
    {

        class Output : public SinkOp
        {

        public:

            Output(const string &name);
            bool setInputTensor(Data::TensorIterator &tensor, byte_type idx);
            Tensor getOutputDef(byte_type);
            
        };

    }

}

#endif // OUTPUT_HPP_