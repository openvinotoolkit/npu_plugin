#include "include/mcm/base/json/number_integer.hpp"

mv::json::NumberInteger::NumberInteger(Object& owner, const std::string& key, int value) :
Value(owner, key, JSONType::NumberInteger),
value_(value)
{

}

mv::json::NumberInteger::operator int&()
{
    return value_;
}

std::string mv::json::NumberInteger::stringify() const
{
    std::ostringstream ss;
    ss << value_;
    return ss.str();
}