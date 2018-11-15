#include "include/mcm/pass/pass_entry.hpp"

mv::pass::PassEntry::PassEntry(const std::string& name) :
name_(name)
{

}

mv::pass::PassEntry& mv::pass::PassEntry::setGenre(PassGenre passGenre)
{
    if (passGenre_.find(passGenre) != passGenre_.end())
        throw MasterError(*this, "Duplicated pass genre definition");
    passGenre_.insert(passGenre);
    return *this;
}

mv::pass::PassEntry& mv::pass::PassEntry::setGenre(const std::initializer_list<PassGenre> &passGenres)
{
    for (auto it = passGenres.begin(); it != passGenres.end(); ++it)
        if (passGenre_.find(*it) != passGenre_.end())
            throw MasterError(*this, "Duplicated pass genre definition");
    passGenre_.insert(passGenres);
    return *this;
}

mv::pass::PassEntry& mv::pass::PassEntry::setDescription(const std::string& description)
{
    description_ = description;
    return *this;
}

mv::pass::PassEntry& mv::pass::PassEntry::setFunc(const std::function<void(const PassEntry&, ComputationModel&, TargetDescriptor&, 
    json::Object&, json::Object&)>& passFunc)
{
    passFunc_ = passFunc;
    return *this;
}

const std::string mv::pass::PassEntry::getName() const
{
    return name_;
}

const std::set<mv::PassGenre> mv::pass::PassEntry::getGenre() const
{
    return passGenre_;
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

const std::map<std::string, mv::json::JSONType>& mv::pass::PassEntry::getArgs() const
{
    return requiredArgs_;
}

std::size_t mv::pass::PassEntry::argsCount() const
{
    return requiredArgs_.size();
}

void mv::pass::PassEntry::run(ComputationModel& model, TargetDescriptor& targetDescriptor, json::Object& compDescriptor, json::Object& output) const
{
    passFunc_(*this, model, targetDescriptor, compDescriptor, output);
}

std::string mv::pass::PassEntry::getLogID() const
{
    return "Pass:" + getName();
}