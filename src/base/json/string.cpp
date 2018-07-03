#include "include/mcm/base/json/string.hpp"

mv::json::String::String(const std::string& value) :
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