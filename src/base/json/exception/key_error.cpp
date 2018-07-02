#include "include/mcm/base/json/exception/key_error.hpp"

mv::json::KeyError::KeyError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}