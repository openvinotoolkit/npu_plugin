#include "include/mcm/base/json/exception/value_error.hpp"

mv::json::ValueError::ValueError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}