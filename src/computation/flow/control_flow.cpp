#include "include/mcm/computation/flow/control_flow.hpp"

mv::ControlFlow::ControlFlow(Control::OpListIterator &source, Control::OpListIterator &sink) :
ComputationFlow("cf_" + source->getName() + "_" + sink->getName())
{
    log(Logger::MessageType::MessageInfo, "Initialized");
    set<std::string>("sourceOp", source->getName());
    set<std::string>("sinkOp", sink->getName());
}

mv::ControlFlow::~ControlFlow()
{
    log(Logger::MessageType::MessageInfo, "Deleted");
}

/*mv::ControlFlow::ControlFlow(mv::json::Value &value):
ComputationFlow(value)
{

}*/

std::string mv::ControlFlow::toString() const
{
    return "control flow '" + name_ + "' " + Element::attrsToString_();
}

/*mv::json::Value mv::ControlFlow::toJsonValue() const
{
    mv::json::Value toReturn = mv::ComputationElement::toJsonValue();
    //toReturn["type"] = mv::Jsonable::toJsonValue("control_flow");
    return toReturn;
}*/
