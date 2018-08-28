#include "include/mcm/computation/op/def/output.hpp"

mv::op::Output::Output(const std::string &name) : 
ComputationOp(OpType::Output, name),
SinkOp(OpType::Output, 1, name)
{
    addAttr("executable", AttrType::BoolType, false);
}

mv::op::Output::Output(mv::json::Value& obj) :
ComputationOp(obj),
SinkOp(obj)
{

}

bool mv::op::Output::setInputTensor(Data::TensorIterator &tensor, std::size_t idx)
{

    bool result = SinkOp::setInputTensor(tensor, idx);
    if (result)
    {
        if (!hasAttr("shape"))
            addAttr("shape", AttrType::ShapeType, tensor->getShape());
        else
            getAttr("shape").setContent<Shape>(tensor->getShape());       
    }
    return result;

}

mv::Tensor mv::op::Output::getOutputDef(std::size_t)
{
    logger_.log(Logger::MessageType::MessageWarning, "Attempt of getting output tensor of model output operation");
    return Tensor();
}


bool mv::op::Output::isHardwarizeable(json::Object&)
{
    return false;
}
