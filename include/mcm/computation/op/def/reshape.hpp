#ifndef RESHAPE_HPP_
#define RESHAPE_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{

    namespace op
    {

        class Reshape : public SourceOp, public SinkOp
        {

        public:

            Reshape(Shape outputShape, const string& name);
            Tensor getOutputDef(byte_type idx);

        };

    }

}

#endif // RESHAPE_HPP_