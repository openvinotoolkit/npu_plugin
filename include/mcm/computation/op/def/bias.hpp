#ifndef BIAS_OP_HPP_
#define BIAS_OP_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)   
        class Bias : public SourceOp, public SinkOp
        {

        public:

            Bias(const std::string& name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(json::Object& targetDescriptor);

        };

    }

}

#endif // BIAS_OP_HPP_
