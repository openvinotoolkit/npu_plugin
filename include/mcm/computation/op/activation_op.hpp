#ifndef ACTIVATION_OP_HPP_
#define ACTIVATION_OP_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{

    class ActivationOp : public SourceOp, public SinkOp
    {

    public:

        ActivationOp(OpType activationType, const string& name);
        ActivationOp(mv::json::Value& value);
        virtual ~ActivationOp() = 0;
        Tensor getOutputDef(byte_type idx);

    };

}

#endif // ACTIVATION_OP_HPP_
