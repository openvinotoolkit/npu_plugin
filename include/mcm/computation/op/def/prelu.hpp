#ifndef PRELU_HPP_
#define PRELU_HPP_

#include "include/mcm/computation/op/activation_op.hpp"

namespace mv
{

    namespace op
    {

        class PReLU : public SourceOp, public SinkOp
        {

        public:

            PReLU(const string &name);
            PReLU(mv::json::Value &obj);
            Tensor getOutputDef(byte_type idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // PRELU_HPP_
