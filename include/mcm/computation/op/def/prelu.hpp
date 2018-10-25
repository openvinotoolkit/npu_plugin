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

            PReLU(const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);
            void gatherSerialFields() override;
        };

    }

}

#endif // PRELU_HPP_
