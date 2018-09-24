#include "include/mcm/base/json/object.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/null.hpp"
#include "include/mcm/base/json/array.hpp"

mv::json::Object::Object()
{
    
}

mv::json::Object::Object(const Object& other) :
members_(other.members_)
{

}

mv::json::Object::Object(std::initializer_list<std::pair<const std::string, Value>> l) :
members_(l)
{

}

void mv::json::Object::emplace(const std::string& key, const Value& value)
{
    members_.emplace(key, value);
}

void mv::json::Object::erase(const std::string& key)
{
    members_.erase(key);
}

unsigned mv::json::Object::size() const
{
    return members_.size();
}

void mv::json::Object::clear()
{
    members_.clear();
}

bool mv::json::Object::hasKey(const std::string& key) const
{
    if (members_.find(key) != members_.end())
        return true;

    return false;
}

std::vector<std::string> mv::json::Object::getKeys() const
{
    std::vector<std::string> keys;

    for (auto it = members_.begin(); it != members_.end(); ++it)
        keys.push_back(it->first);

    return keys;
}

bool mv::json::Object::operator==(const Object& other) const
{
    if (size() != other.size())
        return false;

    auto e1 = members_.begin();
    for (auto e2 = other.members_.begin(); e2 != other.members_.end(); ++e2)
    {
        if (*e1 != *e2)
            return false;
        ++e1;
    }

    return true;
}

bool mv::json::Object::operator!=(const Object& other) const
{
    return !operator==(other);
}

std::string mv::json::Object::stringify() const
{

    std::string output = "{";

    auto it = members_.begin();

    if (it != members_.end())
    {

        auto str = [&it](){ return "\"" + it->first + "\":" + it->second.stringify(); };
        output += str();
        ++it;

        for (; it != members_.end(); ++it)
            output += "," + str();

    }

    output += "}";
    return output;
    
}

std::string mv::json::Object::stringifyPretty() const
{

    std::string output = "{\n";

    auto it = members_.begin();

    if (it != members_.end())
    {

        auto str = [&it]()
        { 
            std::string s = "\"" + it->first + "\": ";
            if (it->second.valueType() == JSONType::Array)
                s += "\n" + it->second.get<Array>().stringifyPretty();
            else if (it->second.valueType() == JSONType::Object)
                s += "\n" + it->second.get<Object>().stringifyPretty();
            else
                s += it->second.stringify();

            for (std::size_t i = 0; i < s.size(); ++i)
            {
                if (s[i] == '\n')
                    s.insert(i + 1, "\t");
            }

            return s;
        };

        output += "\t" + str();
        ++it;

        for (; it != members_.end(); ++it)
            output += ",\n\t" + str();

    }

    output += "\n}";
    return output;

}

mv::json::Value& mv::json::Object::operator[](const std::string& key)
{

    if (members_.find(key) == members_.end())
        emplace(key, Value());

    return members_[key];

}

const mv::json::Value& mv::json::Object::operator[](const std::string& key) const
{
    if (members_.find(key) == members_.end())
        throw ArgumentError(*this, "key", key, "Not a memeber of the object");
    return members_.at(key);
}

mv::json::Object& mv::json::Object::operator=(const Object& other)
{
    members_ = other.members_;
    return *this;
}

std::string mv::json::Object::getLogID() const
{
    return "json::Object (" + std::to_string(size()) + " members)";
}