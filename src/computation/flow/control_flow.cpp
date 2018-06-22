#include "include/mcm/computation/flow/control_flow.hpp"

mv::ControlFlow::ControlFlow(Control::OpListIterator &source, Control::OpListIterator &sink) :
ComputationFlow("cf_" + source->getName() + "_" + sink->getName())
{

}

mv::string mv::ControlFlow::toString() const
{
    return "control flow '" + name_ + "' " + ComputationElement::toString();
}