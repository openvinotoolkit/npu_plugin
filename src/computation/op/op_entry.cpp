#include "include/mcm/computation/op/op_entry.hpp"
#include "include/mcm/computation/op/op_registry.hpp"

mv::op::OpEntry::OpEntry(const std::string& opType) :
opType_(opType)
{

}

mv::op::OpEntry& mv::op::OpEntry::setInputs(std::vector<std::string> labels)
{

    inputLabels_ = labels;
    return *this;

}

mv::op::OpEntry& mv::op::OpEntry::setOutputs(std::vector<std::string> labels)
{

    outputLabels_ = labels;
    return *this;

}

mv::op::OpEntry& mv::op::OpEntry::setInputCheck(const std::function<std::pair<bool, std::size_t>
    (const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::string&)>& inputCheck)
{
    inputCheck_ = inputCheck;
    return *this;
}

mv::op::OpEntry& mv::op::OpEntry::setOutputDef(std::function<void(const std::vector<Data::TensorIterator>&,
    const std::map<std::string, Attribute>&, std::vector<Tensor>&)>& outputDef)
{
    outputDef_ = outputDef;
    return *this;
}

mv::op::OpEntry& mv::op::OpEntry::setDescription(const std::string& description)
{
    description_ = description;
    return *this;
}

mv::op::OpEntry& mv::op::OpEntry::setTypeTrait(const std::string& trait)
{
    
    if (!OpRegistry::checkTypeTrait(trait))
        throw OpError(*this, "Attempt of setting an illegal op type trait " + trait);

    if (typeTraits_.find(trait) == typeTraits_.end())
        typeTraits_.insert(trait);
    return *this;

}

mv::op::OpEntry& mv::op::OpEntry::setTypeTrait(std::initializer_list<std::string> traits)
{

    for (auto it = traits.begin(); it != traits.end(); ++it)
        setTypeTrait(*it);
    return *this;

}

const std::string mv::op::OpEntry::getDescription() const
{
    return description_;
}

std::size_t mv::op::OpEntry::getInputsCount() const
{
    return inputLabels_.size();
}

std::size_t mv::op::OpEntry::getOutputsCount() const
{
    return outputLabels_.size();
}

bool mv::op::OpEntry::hasArg(const std::string& name) const
{
    return std::find_if(args_.begin(), args_.end(),
        [&name](std::tuple<std::string, std::type_index, Attribute> arg)->bool
        {
            return std::get<0>(arg) == name;
        }
    ) != args_.end();
}

std::type_index mv::op::OpEntry::argType(const std::string& name) const
{
    if (!hasArg(name))
        throw OpError(*this, "Attempt of checking the type of an non-existing argument \"" + name + "\"");
    
    return std::get<1>(*std::find_if(args_.begin(), args_.end(),
        [&name](std::tuple<std::string, std::type_index, Attribute> arg)->bool
        {
            return std::get<0>(arg) == name;
        }
    ));
}

std::vector<std::string> mv::op::OpEntry::argsList() const
{
    std::vector<std::string> list;
    list.reserve((args_.size()));
    for (auto &arg : args_)
        list.push_back(std::get<0>(arg));
    return list;
}

std::vector<std::pair<std::string, mv::Attribute>> mv::op::OpEntry::argsListWithDefaultValues() const
{
    std::vector<std::pair<std::string, Attribute>> list;
    list.reserve((args_.size()));
    
    std::for_each(args_.begin(), args_.end(),
        [&list](std::tuple<std::string, std::type_index, Attribute> arg)
        {
            if (std::get<2>(arg).valid())
                list.push_back(make_pair(std::get<0>(arg),std::get<2>(arg)));
        }
    );
    return list;
}

std::pair<bool, std::size_t> mv::op::OpEntry::checkInputs(const std::vector<Data::TensorIterator>& inputs, 
    const std::map<std::string, Attribute>& args, std::string& errMsg)
{
    return inputCheck_(inputs, args, errMsg);
}

void mv::op::OpEntry::getOutputsDef(const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
    std::vector<Tensor>& outputs)
{
    outputDef_(inputs, args, outputs);
}

std::string mv::op::OpEntry::getInputLabel(std::size_t idx)
{

    if (idx >= inputLabels_.size())
        throw IndexError(*this, idx, "Passed input index exceeds inputs count registered for the op type " + opType_);

    return inputLabels_[idx];

}

const std::vector<std::string>& mv::op::OpEntry::getInputLabel()
{
    return inputLabels_;
}

std::string mv::op::OpEntry::getOutputLabel(std::size_t idx)
{

    if (idx >= outputLabels_.size())
        throw IndexError(*this, idx, "Passed input index exceeds outputs count registered for the op type " + opType_);

    return outputLabels_[idx];

}

const std::vector<std::string>& mv::op::OpEntry::getOutputLabel()
{
    return outputLabels_;
}

bool mv::op::OpEntry::hasTypeTrait(const std::string& trait)
{
    return typeTraits_.find(trait) != typeTraits_.end();
}

const std::set<std::string>& mv::op::OpEntry::getTypeTraits()
{
    return typeTraits_;
}

std::string mv::op::OpEntry::getLogID() const
{
    return "OpEntry:" + opType_;
}