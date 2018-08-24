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

            MatMul(const string &name);
            MatMul(mv::json::Value &obj);

            Tensor getOutputDef(byte_type idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // MAT_MUL_HPP_
