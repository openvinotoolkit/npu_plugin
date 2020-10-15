#include "include/mcm/pass/pass_entry.hpp"

mv::pass::PassEntry::PassEntry(const std::string& name) :
name_(name)
{

}

mv::pass::PassEntry& mv::pass::PassEntry::setDescription(const std::string& description)
{
    description_ = description;
    return *this;
}

mv::pass::PassEntry& mv::pass::PassEntry::setFunc(const std::function<void(const PassEntry&, ComputationModel&, TargetDescriptor&, 
    Element&, Element&)>& passFunc)
{
    passFunc_ = passFunc;
    return *this;
}

const std::string mv::pass::PassEntry::getName() const
{
    return name_;
}

const std::string mv::pass::PassEntry::getDescription() const
{
    return description_;
}

mv::pass::PassEntry& mv::pass::PassEntry::defineArg(json::JSONType argType, std::string argName)
{
    if (requiredArgs_.find(argName) != requiredArgs_.end())
        throw MasterError(*this, "Duplicated pass argument definition");
    requiredArgs_.emplace(argName, argType);
    return *this;
}

mv::pass::PassEntry& mv::pass::PassEntry::setLabel(const std::string& label)
{
    labels_.emplace(label);
    return *this;
}

const std::map<std::string, mv::json::JSONType>& mv::pass::PassEntry::getArgs() const
{
    return requiredArgs_;
}

std::size_t mv::pass::PassEntry::argsCount() const
{
    return requiredArgs_.size();
}

bool mv::pass::PassEntry::hasLabel(const std::string& label) const
{
    return labels_.find(label) != labels_.end();
}

void mv::pass::PassEntry::run(ComputationModel& model, TargetDescriptor& targetDescriptor, Element& passDescriptor, Element& output) const
{
    passFunc_(*this, model, targetDescriptor, passDescriptor, output);
}

std::string mv::pass::PassEntry::getLogID() const
{
    return "Pass:" + getName();
}