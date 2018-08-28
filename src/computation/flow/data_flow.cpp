#include "include/mcm/computation/flow/data_flow.hpp"

mv::DataFlow::DataFlow(const Data::OpListIterator& source, std::size_t outputIdx, const Data::OpListIterator& sink, 
    std::size_t inputIdx, const Data::TensorIterator& data) :
ComputationFlow("df_" + source->getName() + Printable::toString(outputIdx) + "_" + sink->getName() + Printable::toString(inputIdx)),
data_(data)
{
    addAttr("sourceOp", AttrType::StringType, source->getName());
    addAttr("sourceOutput", AttrType::ByteType, outputIdx);
    addAttr("sinkOp", AttrType::StringType, sink->getName());
    addAttr("sinkInput", AttrType::ByteType, inputIdx);
}

mv::DataFlow::DataFlow(mv::json::Value &value):
ComputationFlow(value)
{

}

mv::DataFlow::DataFlow(mv::json::Value &value, const Data::TensorIterator& data):
ComputationFlow(value),
data_(data)
{

}

mv::Data::TensorIterator& mv::DataFlow::getTensor()
{
    return data_;
}

std::string mv::DataFlow::toString() const
{
    return "data flow '" + name_ + "'\n'tensor': " + data_->getName() + ComputationElement::toString();
}

mv::json::Value mv::DataFlow::toJsonValue() const
{
    mv::json::Value toReturn = mv::ComputationElement::toJsonValue();
    toReturn["tensor"] = data_->getName();
    //toReturn["type"] = mv::Jsonable::toJsonValue("data_flow");
    return mv::json::Value(toReturn);
}
