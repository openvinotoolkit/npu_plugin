#include "include/fathom/computation/model/element.hpp"

mv::allocator mv::ComputationElement::allocator_;
mv::Attribute mv::ComputationElement::unknownAttr_;

mv::ComputationElement::ComputationElement(const Logger &logger, const string &name) : 
logger_(logger),
name_(name),
attributes_(allocator_)
{

}
mv::ComputationElement::ComputationElement(const ComputationElement &other) :
logger_(other.logger_),
name_(other.name_),
attributes_(allocator_)
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

bool mv::ComputationElement::hasAttr(const string &name)
{
    
    if (attributes_.find(name) != attributes_.end())
        return true;

    return false;

}

mv::Attribute& mv::ComputationElement::getAttr(const string &name)
{
    if (attributes_.find(name) != attributes_.end())
        return attributes_[name];
    else
        return unknownAttr_;

}

mv::vector<mv::string> mv::ComputationElement::getAttrKeys() const
{
    mv::vector<mv::string> attrKeys;
    for (auto it = attributes_.cbegin(); it != attributes_.cend(); ++it)
        attrKeys.push_back(it->first);
    return attrKeys;
}

mv::AttrType mv::ComputationElement::getAttrType(const string &name)
{
    if (attributes_.find(name) != attributes_.end())
        return attributes_[name].getType();
    else
        return AttrType::UnknownType;
}

mv::unsigned_type mv::ComputationElement::attrsCount() const
{
    return attributes_.size();
}

mv::string mv::ComputationElement::toString() const
{
    string result;

    for (auto it = attributes_.cbegin(); it != attributes_.cend(); ++it)
        result += "\n'" +  it->first + "' " + it->second.toString();

    return result;

}