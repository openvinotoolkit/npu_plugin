#include "include/mcm/base/json/string.hpp"

mv::json::String::String(const std::string& value) :
Value(JSONType::String),
value_(value)
{

}

mv::json::String::operator std::string() const
{
    return value_;
}