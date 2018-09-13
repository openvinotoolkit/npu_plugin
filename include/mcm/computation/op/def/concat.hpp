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

            Concat(const std::string &name);
            Concat(mv::json::Value &obj);

            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}

#endif // CONCAT_HPP_
