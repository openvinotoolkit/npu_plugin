#include "include/mcm/base/json/number_float.hpp"

mv::json::NumberFloat::NumberFloat(Object& owner, const std::string& key, float value) :
Value(owner, key, JSONType::NumberFloat),
value_(value)
{

}

mv::json::NumberFloat::operator float&()
{
    return value_;
}

std::string mv::json::NumberFloat::stringify() const
{
    std::ostringstream ss;
    ss << value_;
    std::string output = ss.str();
    if (output.find('.') == output.npos)
        output += ".0";
    return output;
}