#ifndef CONCAT_HPP_
#define CONCAT_HPP_

#include "include/mcm/computation/op/sink_op.hpp"
#include "include/mcm/computation/op/source_op.hpp"


namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)   
        class Concat : public SinkOp, public SourceOp
        {

        public:

            Concat(const string &name);
            Concat(mv::json::Value &obj);

            Tensor getOutputDef(byte_type idx);

        };

    }

}

#endif // CONCAT_HPP_
