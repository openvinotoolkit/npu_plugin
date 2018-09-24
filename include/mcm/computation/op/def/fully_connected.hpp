#ifndef FULLY_CONNECTED_HPP_
#define FULLY_CONNECTED_HPP_

#include "include/mcm/computation/op/sink_op.hpp"
#include "include/mcm/computation/op/source_op.hpp"


namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)   
        class FullyConnected : public SinkOp, public SourceOp
        {

        public:

            FullyConnected(const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}

#endif // FULLY_CONNECTED_HPP_
