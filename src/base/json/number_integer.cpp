#include "include/mcm/base/json/number_integer.hpp"

mv::json::NumberInteger::NumberInteger(long long value) :
value_(value)
{

}

mv::json::NumberInteger::operator long long&()
{
    return value_;
}

std::string mv::json::NumberInteger::stringify() const
{
    std::ostringstream ss;
    ss << value_;
    return ss.str();
}

std::string mv::json::NumberInteger::stringifyPretty() const
{
    return stringify();
}
