#include "include/mcm/utils/parser/exception/parsing_error.hpp"

mv::ParsingError::ParsingError(const std::string& whatArg) :
std::runtime_error(whatArg)
{

}
