#ifndef MAT_MUL_HPP_
#define MAT_MUL_HPP_

#include "include/mcm/computation/op/sink_op.hpp"
#include "include/mcm/computation/op/source_op.hpp"


namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)
        class MatMul : public SinkOp, public SourceOp
        {

        public:

            MatMul(const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);
            void gatherSerialFields() override;
        };

    }

}

#endif // MAT_MUL_HPP_
