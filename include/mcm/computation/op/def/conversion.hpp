#ifndef CONVERSION_H
#define CONVERSION_H

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{

    namespace op
    {

        class Conversion : public SinkOp, public SourceOp
        {

        public:

            Conversion(const string &name, Order targetOrder);
            Conversion(mv::json::Value &obj);

            Tensor getOutputDef(byte_type idx);
            bool isHardwarizeable(mv::json::Object& TargetDescriptor);

        };

    }

}

#endif // CONVERSION_H
