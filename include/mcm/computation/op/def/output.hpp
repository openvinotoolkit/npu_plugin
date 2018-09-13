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

            Output(const std::string &name);
            Output(mv::json::Value &obj);

            bool setInputTensor(Data::TensorIterator &tensor, std::size_t idx);
            Tensor getOutputDef(std::size_t);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}

#endif // OUTPUT_HPP_
