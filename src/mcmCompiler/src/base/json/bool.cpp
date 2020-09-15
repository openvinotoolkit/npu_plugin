#include "include/mcm/base/json/bool.hpp"

mv::json::Bool::Bool(bool value) :
value_(value)
{

}

mv::json::Bool::operator bool&()
{
    return value_;
}

bool mv::json::Bool::operator==(const Bool& other) const
{
    return value_ == other.value_;
}

bool mv::json::Bool::operator!=(const Bool& other) const
{
    return !operator==(other);
}

std::string mv::json::Bool::stringify() const
{
    if (value_)
        return "true";
    
    return "false";
}

std::string mv::json::Bool::stringifyPretty() const
{
    return stringify();
}

std::string mv::json::Bool::getLogID() const
{
    return "json::Bool ("+ stringify() + ")";
}