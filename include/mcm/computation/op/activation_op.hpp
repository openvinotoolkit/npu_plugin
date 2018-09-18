#ifndef ACTIVATION_OP_HPP_
#define ACTIVATION_OP_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{

    class ActivationOp : public SourceOp, public SinkOp
    {

    public:

        ActivationOp(OpType activationType, const std::string& name);
        virtual ~ActivationOp() = 0;
        Tensor getOutputDef(std::size_t idx);

    };

}

#endif // ACTIVATION_OP_HPP_
