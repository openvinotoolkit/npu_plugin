#include "include/mcm/base/json/number_integer.hpp"

mv::json::NumberInteger::NumberInteger(int value) :
Value(JSONType::NumberInteger),
value_(value)
{

}

mv::json::NumberInteger::operator int() const
{
    return value_;
}