#include "include/mcm/base/element.hpp"



mv::Element::Element(const std::string &name) :
name_(name)
{

}

mv::Element::Element(const Element &other) :
attrs_(other.attrs_),
name_(other.name_)
{

}

mv::Element::Element(const std::string& name, const json::Value& content) :
name_(name)
{
    if (content.valueType() != json::JSONType::Object)
        throw AttributeError(*this, "Unable to construct using non json::Object "
            "value type " + json::Value::typeName(content.valueType()));
    auto keys = content.getKeys();
    for (auto const &key : keys)
    {
        attrs_.emplace(key, content[key]);
    }
}

mv::Element::~Element()
{

}

std::string mv::Element::getLogID() const
{
    return "Element '" + name_ + "'";
}

mv::Element& mv::Element::operator=(const Element &other)
{
    name_ = other.name_;
    return *this;
}

const std::string& mv::Element::getName() const
{
    return name_;
}

void mv::Element::setName(const std::string& name)
{
    name_ = name;
}

bool mv::Element::hasAttr(const std::string &name) const
{
    return attrs_.find(name) != attrs_.end();
}

std::size_t mv::Element::attrsCount() const
{
    return attrs_.size();
}

std::vector<std::string> mv::Element::attrsKeys() const
{
    std::vector<std::string> output;
    for (auto &attr : attrs_)
        output.push_back(attr.first);
    return output;
}

void mv::Element::clear()
{
    attrs_.clear();
}

std::string mv::Element::attrsToString_() const
{

    std::string result;
    for (auto it = attrs_.cbegin(); it != attrs_.cend(); ++it)
        result += "\n'" +  it->first + "' (" + it->second.getTypeName() + "): " + it->second.toString();
    return result;

}

void mv::Element::set(const std::string& name, const Attribute& attr)
{
    attrs_.emplace(name, attr);
}

std::string mv::Element::toString() const
{
    return getLogID() + attrsToString_();
}

mv::json::Value mv::Element::toJSON() const
{
    
    json::Object result;
    for (auto it = attrs_.cbegin(); it != attrs_.cend(); ++it)
        result.emplace(it->first, it->second.toJSON());
    return result;

}