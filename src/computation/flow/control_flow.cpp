#include "include/mcm/computation/flow/control_flow.hpp"

mv::ControlFlow::ControlFlow(Control::OpListIterator &source, Control::OpListIterator &sink) :
ComputationFlow("cf_" + source->getName() + "_" + sink->getName())
{
    addAttr("sourceOp", AttrType::StringType, source->getName());
    addAttr("sinkOp", AttrType::StringType, sink->getName());
}

mv::string mv::ControlFlow::toString() const
{
    return "control flow '" + name_ + "' " + ComputationElement::toString();
}

mv::json::Value mv::ControlFlow::toJsonValue() const
{
    mv::json::Value toReturn = mv::ComputationElement::toJsonValue();
    toReturn["type"] = mv::Jsonable::toJsonValue("control_flow");
    return toReturn;
}
