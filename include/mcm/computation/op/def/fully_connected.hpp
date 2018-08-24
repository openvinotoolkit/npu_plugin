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

            FullyConnected(const string &name);
            FullyConnected(mv::json::Value &obj);

            Tensor getOutputDef(byte_type idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // FULLY_CONNECTED_HPP_
