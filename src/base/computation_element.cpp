#include "include/mcm/base/computation_element.hpp"

mv::ComputationElement::ComputationElement(const std::string &name) : 
name_(name)
{

}

mv::ComputationElement::ComputationElement(mv::json::Value& value)
{

    name_ = mv::Jsonable::constructStringFromJson(value["name"]);
    mv::json::Value attributes = value["attributes"];
    std::vector<std::string> keys(attributes.getKeys());
    for(std::size_t i = 0; i < keys.size(); ++i)
        addAttr(keys[i], mv::Attribute::JsonAttributeFactory(attributes[keys[i]]));

}

mv::ComputationElement::ComputationElement(const ComputationElement &other) :
name_(other.name_)
{

    for (auto it = other.attributes_.cbegin(); it != other.attributes_.cend(); ++it)
        attributes_[it->first] = it->second;

}

mv::ComputationElement& mv::ComputationElement::operator=(const ComputationElement &other)
{

    this->name_ = other.name_;
    
    for (auto it = other.attributes_.cbegin(); it != other.attributes_.cend(); ++it)
        attributes_[it->first] = it->second;

    return *this;

}

mv::ComputationElement::~ComputationElement()
{
    
}

const std::string &mv::ComputationElement::getName() const
{
    return name_;
}

void mv::ComputationElement::setName(const std::string& name)
{
    name_ = name;
}

bool mv::ComputationElement::addAttr(const std::string &name, const Attribute &attr)
{

    if (attributes_.find(name) == attributes_.end())
    {
        log(Logger::MessageType::MessageDebug, "Adding attribute '" + name + "' " + attr.toString());
        attributes_[name] = attr;
        return true;
    }
    else
    {
        log(Logger::MessageType::MessageWarning, "Failed adding attribute of a duplicated name '" + name + "'");
        return false;
    }

}

bool mv::ComputationElement::hasAttr(const std::string &name) const
{
    
    if (attributes_.find(name) != attributes_.cend())
        return true;

    return false;

}

mv::Attribute& mv::ComputationElement::getAttr(const std::string &name)
{
    if (attributes_.find(name) == attributes_.end())
        throw ArgumentError("Attribute", name, "Attempt of getting an undefined attribute");

    return attributes_.at(name);

}

const mv::Attribute& mv::ComputationElement::getAttr(const std::string &name) const
{
    if (attributes_.find(name) == attributes_.end())
        throw ArgumentError("Attribute", name, "Attempt of getting an undefined attribute");

    return attributes_.at(name);

}

std::vector<std::string> mv::ComputationElement::getAttrKeys() const
{
    std::vector<std::string> attrKeys;
    for (auto it = attributes_.cbegin(); it != attributes_.cend(); ++it)
        attrKeys.push_back(it->first);
    return attrKeys;
}

mv::AttrType mv::ComputationElement::getAttrType(const std::string &name) const
{
    if (attributes_.find(name) != attributes_.cend())
        return attributes_.at(name).getType();
    else
        return AttrType::UnknownType;
}

std::size_t mv::ComputationElement::attrsCount() const
{
    return attributes_.size();
}

bool mv::ComputationElement::removeAttr(const std::string &name)
{

    auto it = attributes_.find(name);

    if (it != attributes_.end())
    {
        attributes_.erase(it);
        return false;
    }
    
    return false;
}

std::string mv::ComputationElement::toString() const
{
    std::string result;

    for (auto it = attributes_.cbegin(); it != attributes_.cend(); ++it)
        result += "\n'" +  it->first + "' " + it->second.toString();

    return result;

}

mv::json::Value mv::ComputationElement::toJsonValue() const
{
    mv::json::Object obj;
    mv::json::Object attr;

    obj["name"] = mv::Jsonable::toJsonValue(name_);
    for (auto it = attributes_.cbegin(); it != attributes_.cend(); ++it)
        attr[it->first] = mv::Jsonable::toJsonValue(it->second);

    obj["attributes"] = attr;
    return mv::json::Value(obj);

}

bool mv::ComputationElement::operator<(ComputationElement &other)
{
    return name_ < other.name_;
}

bool mv::ComputationElement::operator ==(const ComputationElement& other)
{
    return name_ == other.name_;
}

std::string mv::ComputationElement::getLogID_() const
{
    return "ComputationElement " + getName();
}