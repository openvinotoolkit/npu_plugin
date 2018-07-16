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

bool mv::json::Object::hasKey(const std::string& key)
{
    if (members_.find(key) != members_.end())
        return true;

    return false;
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

mv::json::Value& mv::json::Object::operator[](const std::string& key)
{

    if (members_.find(key) == members_.end())
    {
        emplace(key, Value());
    }

    return members_[key];

}

mv::json::Object& mv::json::Object::operator=(const Object& other)
{
    members_ = other.members_;
    return *this;
}