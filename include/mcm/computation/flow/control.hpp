#ifndef CONTROL_FLOW_HPP_
#define CONTROL_FLOW_HPP_

#include "include/mcm/computation/flow/flow.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class ControlFlow : public ComputationFlow
    {

    public:

        ControlFlow(Control::OpListIterator &source, Control::OpListIterator &sink);
        string toString() const;
    };

}

#endif // CONTROL_FLOW_HPP_