#ifndef BATCH_NORM_HPP_
#define BATCH_NORM_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{
    namespace op
    {

        class BatchNorm : public SourceOp, public SinkOp
        {

        public:

            BatchNorm(float_type varianceEps, const string &name);
            BatchNorm(mv::json::Value& obj);

            Tensor getOutputDef(byte_type idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // BATCH_NORM_HPP_
