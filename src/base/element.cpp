#include "include/mcm/base/element.hpp"

mv::Element::Element(const std::string& name) :
name_(name)
{

}

mv::Element::Element(const char* name) :
name_(name)
{

}

mv::Element::Element(const Element &other) :
attrs_(other.attrs_),
name_(other.name_)
{

}

mv::Element::Element(const json::Value& content)
{
    if (content.valueType() != json::JSONType::Object)
        throw AttributeError(*this, "Unable to construct using non json::Object "
            "value type " + json::Value::typeName(content.valueType()));

    if (!content.hasKey("name"))
        throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
            "does not contain 'name' field");

    if (content["name"].valueType() != json::JSONType::String)
        throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
            "field 'name' is " + json::Value::typeName(content["name"].valueType()) + ", must be json::String");

    if (!content.hasKey("attrs"))
        throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
            "does not contain 'attrs' field");

    if (content["attrs"].valueType() != json::JSONType::Object)
        throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
            "field 'attrs' is " + json::Value::typeName(content["attrs"].valueType()) + ", must be json::Object");

    if (content.size() > 2)
        throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
            "too many members specified - must be 2 ('name' and 'attrs'), has " + std::to_string(content.size()));

    name_ = content["name"].get<std::string>();

    auto keys = content["attrs"].getKeys();
    for (auto const &key : keys)
    {
        auto it = attrs_.emplace(key, content[key]);
        if (!it.second)
            throw RuntimeError(*this, "Unable to emplace a new element in attributes dictionary");

        log(Logger::MessageType::MessageDebug, "Attribute '" + key + "' (" + it.first->second.getTypeName() +
                    ") set to " + it.first->second.toString());

    }
}

mv::Element::~Element()
{

}

std::string mv::Element::getLogID() const
{
    return "Element:" + name_;
}

mv::Element& mv::Element::operator=(const Element &other)
{
    name_ = other.name_;
    attrs_ = other.attrs_;
    return *this;
}

bool mv::Element::operator<(const Element& other) const
{
    return name_ < other.name_;
}

bool mv::Element::operator==(const Element& other) const
{
    return name_ == other.name_;
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
        result += "\n\"" +  it->first + "\" (" + it->second.getTypeName() + "): " + it->second.toString();
    return result;

}

void mv::Element::forceErase_(const std::string& name)
{
    attrs_.erase(name);
}

const std::map<std::string, mv::Attribute>& mv::Element::getAttrs_() const
{
    return attrs_;
}

mv::Attribute& mv::Element::get(const std::string& name)
{
    if (attrs_.find(name) == attrs_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return attrs_[name];
}

void mv::Element::set(const std::string& name, const Attribute& attr)
{
    auto it = attrs_.emplace(name, attr);
    if (!it.second)
        throw RuntimeError(*this, "Unable to emplace a new element in attributes dictionary");
    log(Logger::MessageType::MessageDebug, "Attribute '" + name + "' (" + it.first->second.getTypeName() +
        ") set to " + it.first->second.toString());
}

void mv::Element::erase(const std::string& name)
{
    if (!hasAttr(name))
        throw ArgumentError(*this, "attribute identifer", name,  "Undefined identifier");
    if (attrs_[name].hasTrait("const"))
        throw AttributeError(*this, "Attempt of deletion of a const attribute " + name);
    attrs_.erase(name);
}

std::string mv::Element::toString() const
{
    return getLogID() + attrsToString_();
}

mv::json::Value mv::Element::toJSON() const
{

    json::Object result;
    result["name"] = name_;
    result.emplace("attrs", json::Object());
    for (auto it = attrs_.cbegin(); it != attrs_.cend(); ++it)
        result["attrs"].emplace({it->first, it->second.toJSON()});
    return result;

}
