#include "include/mcm/base/json/bool.hpp"

mv::json::Bool::Bool(Object& owner, const std::string& key, bool value) :
Value(owner, key, JSONType::Bool),
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