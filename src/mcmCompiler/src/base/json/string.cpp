#include "include/mcm/base/json/string.hpp"

mv::json::String::String(const std::string& value) :
value_(value)
{

}

mv::json::String::operator std::string&()
{
    return value_;
}

bool mv::json::String::operator==(const String& other) const
{
    return value_ == other.value_;
}

bool mv::json::String::operator!=(const String& other) const
{
    return !operator==(other);
}


std::string mv::json::String::stringify() const
{
    return "\"" + value_ + "\"";
}

std::string mv::json::String::stringifyPretty() const
{
    return stringify();
}

std::string mv::json::String::getLogID() const
{
    return "json::String ("+ stringify() + ")";
}