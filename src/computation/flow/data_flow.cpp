#include "include/mcm/computation/flow/data_flow.hpp"
#include "include/mcm/computation/model/computation_model.hpp"

mv::DataFlow::DataFlow(ComputationModel& model, Data::OpListIterator source, std::size_t outputIdx, Data::OpListIterator sink, 
    std::size_t inputIdx, Data::TensorIterator data) :
ModelElement(model, "df_" + source->getName() + std::to_string(outputIdx) + "_" + sink->getName() + std::to_string(inputIdx))
{
    log(Logger::MessageType::Debug, "Initialized");
    set<std::string>("sourceOp", source->getName(), {"const"});
    set<std::size_t>("sourceOutput", outputIdx, {"const"});
    set<std::string>("sinkOp", sink->getName(), {"const"});
    set<std::size_t>("sinkInput", inputIdx, {"const"});
    set<std::string>("data", data->getName(), {"const"});
}

mv::DataFlow::~DataFlow()
{
    log(Logger::MessageType::Debug, "Deleted");
}

/*mv::DataFlow::DataFlow(mv::json::Value &value):
ComputationFlow(value)
{

}

mv::DataFlow::DataFlow(mv::json::Value &value, const Data::TensorIterator& data):
ComputationFlow(value),
data_(data)
{

}*/

mv::Data::TensorIterator mv::DataFlow::getTensor()
{
    return getModel_().getTensor(get<std::string>("data"));
}

std::string mv::DataFlow::toString() const
{
    return getLogID() + Element::attrsToString_();
}

/*mv::json::Value mv::DataFlow::toJsonValue() const
{
    mv::json::Value toReturn = mv::ComputationElement::toJsonValue();
    toReturn["tensor"] = data_->getName();
    //toReturn["type"] = mv::Jsonable::toJsonValue("data_flow");
    return mv::json::Value(toReturn);
}*/

std::string mv::DataFlow::getLogID() const
{
    return "DataFlow:" + name_;
}