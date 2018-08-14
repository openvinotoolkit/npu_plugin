#include "include/mcm/base/exception/index_error.hpp"

mv::IndexError::IndexError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}