#include "include/mcm/base/json/exception/index_error.hpp"

mv::json::IndexError::IndexError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}