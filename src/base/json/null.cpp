#include "include/mcm/base/json/null.hpp"

mv::json::Null::Null()
{

}

std::string mv::json::Null::stringify() const
{
    return "null";
}