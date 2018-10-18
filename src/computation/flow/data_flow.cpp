#include "include/mcm/computation/flow/data_flow.hpp"

mv::DataFlow::DataFlow(const Data::OpListIterator& source, std::size_t outputIdx, const Data::OpListIterator& sink, 
    std::size_t inputIdx, const Data::TensorIterator& data) :
ComputationFlow("df_" + source->getName() + std::to_string(outputIdx) + "_" + sink->getName() + std::to_string(inputIdx)),
data_(data)
{
    log(Logger::MessageType::Info, "Initialized");
    set<std::string>("sourceOp", source->getName());
    set<std::size_t>("sourceOutput", outputIdx);
    set<std::string>("sinkOp", sink->getName());
    set<std::size_t>("sinkInput", inputIdx);
}

mv::DataFlow::~DataFlow()
{
    log(Logger::MessageType::Info, "Deleted");
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

mv::Data::TensorIterator& mv::DataFlow::getTensor()
{
    return data_;
}

std::string mv::DataFlow::toString() const
{
    return getLogID() + "\n'tensor': " + data_->getName() + Element::attrsToString_();
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