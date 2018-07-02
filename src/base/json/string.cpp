#include "include/mcm/base/json/string.hpp"

mv::json::String::String(Object& owner, const std::string& key, const std::string& value) :
Value(owner, key, JSONType::String),
value_(value)
{

}

mv::json::String::operator std::string&()
{
    return value_;
}

std::string mv::json::String::stringify() const
{
    return "\"" + value_ + "\"";
}