#ifndef SCALE_HPP_
#define SCALE_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{

    namespace op
    {

        class Scale : public SourceOp, public SinkOp
        {

        public:

            Scale(const string &name);
            Scale(mv::json::Value &obj);

            Tensor getOutputDef(byte_type idx);

        };

    }

}

#endif // BATCH_NORM_HPP_
