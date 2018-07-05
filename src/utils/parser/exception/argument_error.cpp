#include "include/mcm/utils/parser/exception/argument_error.hpp"

mv::ArgumentError::ArgumentError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}
