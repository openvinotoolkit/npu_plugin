#include "include/mcm/computation/op/op.hpp"

mv::Op::Op(ComputationModel& model, const std::string& opType, const std::string& name, 
    const std::vector<Data::TensorIterator>& inputs, std::initializer_list<std::pair<std::string, Attribute>> args) :
ModelElement(model, name),
model_(model)
{

    if (!op::OpRegistry::checkOpType(opType))
        throw ArgumentError(*this, "opType", opType, "Unregistered op type");
    
    inputs_.reserve(op::OpRegistry::getInputsCount(opType));
    outputs_.reserve(op::OpRegistry::getOutputsCount(opType));

    set<std::string>("opType", opType, {"const"});

    auto argList = op::OpRegistry::argsList(opType);

    for (auto it = args.begin(); it != args.end(); ++it)
    {

        auto argIt = std::find(argList.begin(), argList.end(), it->first);
        if (argIt != argList.end())
        {
            if (!op::OpRegistry::checkArgType(opType, it->first, it->second.getTypeID()))
                throw ArgumentError(*this, "arg", it->first, "Invalid argument type, received " +  
                    attr::AttributeRegistry::getTypeName(it->second.getTypeID()) + ", must be " + 
                    attr::AttributeRegistry::getTypeName(op::OpRegistry::argType(opType, it->first)));
            set(it->first, it->second);
            argList.erase(argIt);
        }
        else
            throw ArgumentError(*this, "arg", it->first, "Invalid argument");

    }

    if (!argList.empty())
    {
        std::string list;
        for (std::size_t i = 0; i < argList.size() - 1; ++i)
            list += argList[i] + ", ";
        list += argList.back();
        throw ArgumentError(*this, "arg", list, "Missing arguments");
    }

    if (inputs.size() != op::OpRegistry::getInputsCount(opType))
        throw ArgumentError(*this, "inputs:size", std::to_string(inputs.size()), "Does not match the registered inputs count " +
            std::to_string(op::OpRegistry::getInputsCount(opType)));

    for (auto it = inputs.begin(); it != inputs.end(); ++it)
        inputs_.push_back(*it);

    std::string errMsg;
    auto checkRes = op::OpRegistry::checkInputs(opType, inputs_, getAttrs_(), errMsg);

    if (!checkRes.first)
        throw OpError(*this, "Invalid input " + op::OpRegistry::getInputLabel(opType, checkRes.second) + " (" + 
            std::to_string(checkRes.second) + ") - " + errMsg);

    std::vector<Tensor> outputsDef;
    op::OpRegistry::getOutputsDef(opType, inputs_, getAttrs_(), outputsDef);

    DataModel dm(model_);
    for (std::size_t i = 0; i < outputsDef.size(); ++i)
    {
        outputsDef[i].setName(getName() + outputsDef[i].getName());
        outputs_.push_back(dm.defineTensor(outputsDef[i]));
        outputs_[i]->set<std::string>("sourceOp", getName(), {"const"});
    }

    const std::set<std::string>& typeTraits = op::OpRegistry::getTypeTraits(opType);
    std::vector<std::string> opTraits;
    for (auto it = typeTraits.begin(); it != typeTraits.end(); ++it)
        opTraits.push_back(*it);

    set<std::vector<std::string>>("traits", opTraits);

}

std::string mv::Op::getOpType() const
{
    return get<std::string>("opType");
}

void mv::Op::setInputTensor(Data::TensorIterator tensor, std::size_t idx)
{

    inputs_[idx] = tensor;
    std::string errMsg;
    auto checkRes = op::OpRegistry::checkInputs(getOpType(), inputs_, getAttrs_(), errMsg);

    if (!checkRes.first)
        throw OpError(*this, "Invalid input " + op::OpRegistry::getInputLabel(getOpType(), checkRes.second) + " (" + 
            std::to_string(checkRes.second) + ") - " + errMsg);

    std::vector<Tensor> outputsDef;
    op::OpRegistry::getOutputsDef(getOpType(), inputs_, getAttrs_(), outputsDef);
    DataModel dm(model_);
    for (std::size_t i = 0; i < outputsDef.size(); ++i)
    {
        *outputs_[i] = outputsDef[i];
        outputs_[i]->set<std::string>("sourceOp", getName(), {"const"});
    }
}

mv::Data::TensorIterator mv::Op::getInputTensor(std::size_t idx)
{
    if (idx > inputs_.size())
        throw IndexError(*this, idx, "Exceeds the number of inputs");
    return inputs_[idx];
}

mv::Data::TensorIterator mv::Op::getInputTensor(const std::string& label)
{

    const std::vector<std::string>& labels = op::OpRegistry::getInputLabel(getOpType());

    std::size_t idx = 0;
    auto it = labels.begin();
    for (; it != labels.end(); ++it)
    {
        if (*it == label)
            break;
        ++idx;
    }

    if (it == labels.end())
        throw ArgumentError(*this, "input:label", label, "Label not registered for op type " + getOpType());

    return inputs_[idx];

}

std::vector<mv::Data::TensorIterator> mv::Op::getInputTensor()
{
   return inputs_;
}

mv::Data::TensorIterator mv::Op::getOutputTensor(std::size_t idx)
{
     if (idx > inputs_.size())
        throw IndexError(*this, idx, "Exceeds the number of outputs");
    return outputs_[idx];
}

mv::Data::TensorIterator mv::Op::getOutputTensor(const std::string& label)
{

    const std::vector<std::string>& labels = op::OpRegistry::getOutputLabel(getOpType());

    std::size_t idx = 0;
    auto it = labels.begin();
    for (; it != labels.end(); ++it)
    {
        if (*it == label)
            break;
        ++idx;
    }

    if (it == labels.end())
        throw ArgumentError(*this, "output:label", label, "Label not registered for op type " + getOpType());

    return outputs_[idx];
    
}

std::vector<mv::Data::TensorIterator> mv::Op::getOutputTensor()
{
    return outputs_;
}

std::size_t mv::Op::inputSlots()
{
    return inputs_.size();
}

std::size_t mv::Op::outputSlots()
{
    return outputs_.size();
}

std::string mv::Op::getLogID() const
{
    return "Op:" + name_;
}