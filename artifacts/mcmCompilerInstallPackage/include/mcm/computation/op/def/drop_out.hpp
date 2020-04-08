#ifndef DROP_OUT_HPP_
#define DROP_OUT_HPP_

#include "include/mcm/computation/op/sink_op.hpp"
#include "include/mcm/computation/op/source_op.hpp"


namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)
        class DropOut : public SinkOp, public SourceOp
        {

        public:

            DropOut(const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}



#endif /* DROP_OUT_HPP_ */
