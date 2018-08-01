#include "include/mcm/base/json/bool.hpp"

mv::json::Bool::Bool(bool value) :
value_(value)
{

}

mv::json::Bool::operator bool&()
{
    return value_;
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
