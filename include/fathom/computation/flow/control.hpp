#ifndef CONTROL_FLOW_HPP_
#define CONTROL_FLOW_HPP_

#include "include/fathom/computation/flow/flow.hpp"
#include "include/fathom/computation/model/iterator/control_context.hpp"
#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class ControlFlow : public ComputationFlow
    {

    public:

        ControlFlow(Control::OpListIterator &source, Control::OpListIterator &sink) :
        ComputationFlow("cf_" + source->getName() + "_" + sink->getName())
        {

        }

        string toString() const
        {
            return "control flow '" + name_ + "' " + ComputationElement::toString();
        }
        
    };

}

#endif // CONTROL_FLOW_HPP_