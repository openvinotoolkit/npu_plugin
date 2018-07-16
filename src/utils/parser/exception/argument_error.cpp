#include "include/mcm/utils/parser/exception/argument_error.hpp"

mv::ArgumentError::ArgumentError(const std::string& argName, const std::string& argVal,
    const std::string& whatArg) :
std::runtime_error(whatArg),
argName_(argName),
argVal_(argVal)
{

}

const std::string& mv::ArgumentError::getArgName() const
{
    return argName_;
}

const std::string& mv::ArgumentError::getArgVal() const
{
    return argVal_;
}