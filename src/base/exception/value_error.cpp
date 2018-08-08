#include "include/mcm/base/exception/value_error.hpp"

mv::ValueError::ValueError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}