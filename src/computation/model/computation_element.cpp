#include "include/mcm/computation/model/computation_element.hpp"
#include "include/mcm/computation/model/computation_model.hpp"

mv::allocator mv::ComputationElement::allocator_;
mv::Attribute mv::ComputationElement::unknownAttr_;
mv::Logger &mv::ComputationElement::logger_= mv::ComputationModel::logger();

mv::ComputationElement::ComputationElement(const string &name) : 
name_(name)
{

}

mv::ComputationElement::ComputationElement(mv::json::Value& value)
{

    name_ = mv::Jsonable::constructStringFromJson(value["name"]);
    mv::json::Value attributes = value["attributes"];
    std::vector<std::string> keys(attributes.getKeys());
    for(unsigned_type i = 0; i < keys.size(); ++i)
        addAttr(keys[i], mv::Attribute::JsonAttributeFactory(attributes[keys[i]]));

}

mv::ComputationElement::ComputationElement(const ComputationElement &other) :
name_(other.name_)
{

    for (auto it = other.attributes_.cbegin(); it != other.attributes_.cend(); ++it)
    {
        attributes_[it->first] = it->second;
    }

}

mv::ComputationElement& mv::ComputationElement::operator=(const ComputationElement &other)
{

    this->name_ = other.name_;
    
    for (auto it = other.attributes_.cbegin(); it != other.attributes_.cend(); ++it)
    {
        attributes_[it->first] = it->second;
    } 

    return *this;

}

mv::ComputationElement::~ComputationElement()
{
    
}

const mv::string &mv::ComputationElement::getName() const
{
    return name_;
}

bool mv::ComputationElement::addAttr(const string &name, const Attribute &attr)
{

    if (attributes_.find(name) == attributes_.end())
    {
        logger_.log(Logger::MessageType::MessageDebug, "Element '" + name_ + "' - adding attribute '" + name + "' " + attr.toString());
        attributes_[name] = attr;
        return true;
    }
    else
    {
        logger_.log(Logger::MessageType::MessageWarning, "Element '" + name_ + "' - failed adding attribute of a duplicated name '" + name + "'");
        return false;
    }

}

bool mv::ComputationElement::hasAttr(const string &name) const
{
    
    if (attributes_.find(name) != attributes_.cend())
        return true;

    return false;

}

mv::Attribute& mv::ComputationElement::getAttr(const string &name)
{
    if (attributes_.find(name) != attributes_.end())
        return attributes_.at(name);
    else
        return unknownAttr_;

}

const mv::Attribute& mv::ComputationElement::getAttr(const string &name) const
{
    if (attributes_.find(name) != attributes_.cend())
        return attributes_.at(name);
    else
        return unknownAttr_;

}

mv::allocator::vector<mv::string> mv::ComputationElement::getAttrKeys() const
{
    allocator::vector<mv::string> attrKeys;
    for (auto it = attributes_.cbegin(); it != attributes_.cend(); ++it)
        attrKeys.push_back(it->first);
    return attrKeys;
}

mv::AttrType mv::ComputationElement::getAttrType(const string &name) const
{
    if (attributes_.find(name) != attributes_.cend())
        return attributes_.at(name).getType();
    else
        return AttrType::UnknownType;
}

mv::unsigned_type mv::ComputationElement::attrsCount() const
{
    return attributes_.size();
}

bool mv::ComputationElement::removeAttr(const string &name)
{

    auto it = attributes_.find(name);

    if (it != attributes_.end())
    {
        attributes_.erase(it);
        return false;
    }
    
    return false;
}

mv::string mv::ComputationElement::toString() const
{
    string result;

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
