#include "include/mcm/base/element.hpp"

mv::Element::Element(const std::string& name) :
name_(name)
{

}

mv::Element::Element(const char* name) :
name_(name)
{

}

mv::Element::Element(const Element& other) :
attrs_(other.attrs_),
name_(other.name_)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
}

mv::Element::Element(const json::Value& content, bool simplifiedTyping, std::string name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    if (content.valueType() != json::JSONType::Object)
        throw AttributeError(*this, "Unable to construct using non json::Object "
            "value type " + json::Value::typeName(content.valueType()));

    if (!simplifiedTyping) {

        if (!content.hasKey("name"))
            throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
                "does not contain 'name' field");

        if (content["name"].valueType() != json::JSONType::String)
            throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
                "field 'name' is " + json::Value::typeName(content["name"].valueType()) + ", must be json::String");

        name_ = content["name"].get<std::string>();

        if (!content.hasKey("attrs"))
            throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
                "does not contain 'attrs' field");

        if (content["attrs"].valueType() != json::JSONType::Object)
            throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
                "field 'attrs' is " + json::Value::typeName(content["attrs"].valueType()) + ", must be json::Object");

        if (content.size() > 2)
            throw AttributeError(*this, "Invalid json::Object passed for the construction of the Element - "
                "too many members specified - must be 2 ('name' and 'attrs'), has " + std::to_string(content.size()));

        auto keys = content["attrs"].getKeys();
        for (auto const &key : keys)
        {
            auto it = attrs_.emplace(key, content[key]);
            if (!it.second)
                throw RuntimeError(*this, "Unable to emplace a new element in attributes dictionary");

            log(Logger::MessageType::Debug, "Attribute '" + key + "' (" + it.first->second.getTypeName() +
                        ") set to " + it.first->second.toString());

        }
    }
    else
    {
        if (!content.hasKey("name"))
        {
            if (!name.empty())
                name_ = name;
        }
        else
            name_ = content["name"].get<std::string>();

        auto keys = content.getKeys();

        for (auto const &key : keys)
        {
            if (key != "name")
            {
                mv::Attribute val;
                mv::json::Array jsonArray;
                switch (content[key].valueType())
                {
                    case json::JSONType::Array:
                        jsonArray = content[key].get<mv::json::Array>();

                        json::JSONType tFirst, tNext;
                        if (jsonArray.size() > 0)
                        {
                            tFirst = jsonArray[0].valueType();
                            for (size_t i = 1; i < jsonArray.size(); i++)
                            {
                                tNext = jsonArray[i].valueType();
                                if (tFirst != tNext)
                                {
                                    throw ArgumentError(*this, key + ":type", json::Value::typeName(content[key].valueType()),
                                        "Construction of Element from Json Array of different types not currently supported.");
                                }
                            }

                            switch (tFirst)
                            {
                                case json::JSONType::Array:
                                    throw ArgumentError(*this, key + ":type", json::Value::typeName(content[key].valueType()),
                                        "Construction of Element from Json Array of Json Arrays not currently supported.");
                                case json::JSONType::Bool:
                                    val = mv::attr::AttributeRegistry::getFromJSONFunc(typeid(std::vector<bool>))(content[key]);
                                    break;
                                case json::JSONType::NumberFloat:
                                    val = mv::attr::AttributeRegistry::getFromJSONFunc(typeid(std::vector<double>))(content[key]);
                                    break;
                                case json::JSONType::NumberInteger:
                                    val = mv::attr::AttributeRegistry::getFromJSONFunc(typeid(std::vector<int>))(content[key]);
                                    break;
                                case json::JSONType::Object:
                                    val = mv::attr::AttributeRegistry::getFromSimplifiedJSONFunc(typeid(std::vector<mv::Element>))(content[key]);
                                    break;
                                case json::JSONType::String:
                                    val = mv::attr::AttributeRegistry::getFromJSONFunc(typeid(std::vector<std::string>))(content[key]);
                                    break;
                                default:
                                    break;
                            }
                        }

                        break;

                    case json::JSONType::Bool:
                        val = content[key].get<bool>();
                        break;

                    case json::JSONType::NumberFloat:
                        val = content[key].get<double>();
                        break;

                    case json::JSONType::NumberInteger:
                        // XXX: down conversion here...
                        val = static_cast<int>(content[key].get<long long>());
                        break;

                    case json::JSONType::Object:
                        val = Element(content[key], true, key);
                        break;

                    case json::JSONType::String:
                        val = content[key].get<std::string>();
                        break;

                    default:
                        throw ArgumentError(*this, key + ":type", json::Value::typeName(content[key].valueType()),
                            "Invalid simplified from-JSON conversion due to an invalid type");

                }

                auto it = attrs_.emplace(key, val);
                if (!it.second)
                    throw RuntimeError(*this, "Unable to emplace a new element in attributes dictionary");

                log(Logger::MessageType::Debug, "Attribute '" + key + "' (" + it.first->second.getTypeName() +
                            ") set to " + it.first->second.toString());
            }

        }

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
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
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

std::map<std::string, mv::Attribute> mv::Element::getAttrs(const std::vector<std::string>& forbiddenKeys) const
{
    std::map<std::string, Attribute> toReturn(attrs_);
    for(auto& s: forbiddenKeys)
        if(toReturn.find(s) != toReturn.end())
            toReturn.erase(s);
    return toReturn;
}

std::map<std::string, mv::Attribute> mv::Element::attrsToCopy(const std::vector<std::string>& forbiddenKeys) const
{
    std::map<std::string, Attribute> all(attrs_);
    std::map<std::string, Attribute> toReturn;
    for(auto& s: forbiddenKeys)
        if(all.find(s) != all.end())
            toReturn.emplace(s, all.at(s));
    return toReturn;
}

void mv::Element::setAttrs(const std::map<std::string, Attribute>& attrs)
{
    attrs_.insert(attrs.begin(), attrs.end());
}

mv::Attribute& mv::Element::get(const std::string& name)
{
    if (attrs_.find(name) == attrs_.end())
        throw ArgumentError(*this, "name", name, "Undefined attribute");
    return attrs_[name];
}

void mv::Element::set(const std::string& name, const Attribute& attr)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_BASE)
    auto it = attrs_.emplace(name, attr);
    if (!it.second)
        throw RuntimeError(*this, "Unable to emplace a new element in attributes dictionary");
    log(Logger::MessageType::Debug, "Attribute '" + name + "' (" + it.first->second.getTypeName() +
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
    return toJSON(false);
}

mv::json::Value mv::Element::toJSON(bool simplifiedTyping) const
{
    json::Object result;
    if (name_ != "")
        result["name"] = name_;

    if (!simplifiedTyping)
    {
        result.emplace("attrs", json::Object());
        for (auto it = attrs_.cbegin(); it != attrs_.cend(); ++it)
            result["attrs"].emplace({it->first, it->second.toJSON()});
        return result;
    }

    for (auto it = attrs_.cbegin(); it != attrs_.cend(); ++it)
    {
        auto attrTypeID = it->second.getTypeID();
        if (attrTypeID == attr::AttributeRegistry::getTypeID("Element"))
        {
            result.emplace(it->first, it->second.get<Element>().toJSON(true));
            continue;
        }
        else if (attrTypeID == attr::AttributeRegistry::getTypeID("std::vector<mv::Element>"))
        {
            auto jsonVal = attr::AttributeRegistry::getToSimplifiedJSONFunc(attrTypeID)(it->second);
            result.emplace(it->first, jsonVal);
            continue;
        }

        // if (!attr::AttributeRegistry::hasTypeTrait(attrTypeID, "standardJSON"))
        // {
        //     throw ArgumentError(*this, it->first + ":type", attr::AttributeRegistry::getTypeName(attrTypeID),
        //         "Impossible simplified to-JSON conversion, because type is not a JSON type");
        // }
        auto jsonVal = attr::AttributeRegistry::getToJSONFunc(attrTypeID)(it->second);
        result.emplace(it->first, jsonVal);

    }

    return result;

}
