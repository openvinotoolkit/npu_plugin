#include "include/mcm/base/json/number_float.hpp"

mv::json::NumberFloat::NumberFloat(double value) :
value_(value)
{

}

mv::json::NumberFloat::operator double&()
{
    return value_;
}

bool mv::json::NumberFloat::operator==(const NumberFloat& other) const
{
    return value_ == other.value_;
}

bool mv::json::NumberFloat::operator!=(const NumberFloat& other) const
{
    return !operator==(other);
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

std::string mv::json::NumberFloat::stringifyPretty() const
{
    return stringify();
}

std::string mv::json::NumberFloat::getLogID() const
{
    return "json::NumberFloat ("+ stringify() + ")";
}