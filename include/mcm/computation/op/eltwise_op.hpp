#ifndef ELTWISE_OP_HPP_
#define ELTWISE_OP_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{

    class EltwiseOp : public SourceOp, public SinkOp
    {

    public:

        EltwiseOp(OpType eltwiseType, const string &name);
        EltwiseOp(mv::json::Value& value);
        virtual ~EltwiseOp() = 0;
        Tensor getOutputDef(byte_type idx);

    };

}

#endif // ELTWISE_OP_HPP_
