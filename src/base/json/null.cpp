#include "include/mcm/base/json/null.hpp"

mv::json::Null::Null(Object& owner, const std::string& key) :
Value(owner, key, JSONType::Null)
{

}

std::string mv::json::Null::stringify() const
{
    return "null";
}