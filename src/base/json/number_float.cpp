#include "include/mcm/base/json/number_float.hpp"

mv::json::NumberFloat::NumberFloat(float value) :
Value(JSONType::NumberFloat),
value_(value)
{

}

mv::json::NumberFloat::operator float() const
{
    return value_;
}