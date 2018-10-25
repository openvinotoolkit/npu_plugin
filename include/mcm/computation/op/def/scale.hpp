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

            Scale(const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);
            void gatherSerialFields() override;

        };

    }

}

#endif // BATCH_NORM_HPP_
