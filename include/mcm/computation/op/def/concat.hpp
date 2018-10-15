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
            unsigned active_inputs;

        public:

            Concat(const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

            bool hasInputDef() override;
            bool hasInputDef(std::size_t idx) override;

        };

    }

}

#endif // CONCAT_HPP_
