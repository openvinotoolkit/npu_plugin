#include "include/mcm/computation/flow/control_flow.hpp"

mv::ControlFlow::ControlFlow(ComputationModel& model, detailControlFlow::OpListIterator source, detailControlFlow::OpListIterator sink) :
ModelElement(model, "cf_" + source->getName() + "_" + sink->getName())
{
    log(Logger::MessageType::Debug, "Initialized");
    set<std::string>("sourceOp", source->getName(), {"const"});
    set<std::string>("sinkOp", sink->getName(), {"const"});
}

mv::ControlFlow::~ControlFlow()
{
    log(Logger::MessageType::Debug, "Deleted");
}

/*mv::ControlFlow::ControlFlow(mv::json::Value &value):
ComputationFlow(value)
{

}*/

std::string mv::ControlFlow::toString() const
{
    return getLogID() + Element::attrsToString_();
}

/*mv::json::Value mv::ControlFlow::toJsonValue() const
{
    mv::json::Value toReturn = mv::ComputationElement::toJsonValue();
    //toReturn["type"] = mv::Jsonable::toJsonValue("control_flow");
    return toReturn;
}*/

std::string mv::ControlFlow::getLogID() const
{
    return "ControlFlow:" + name_;
}