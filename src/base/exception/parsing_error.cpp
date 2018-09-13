#include "include/mcm/base/exception/parsing_error.hpp"

mv::ParsingError::ParsingError(const LogSender& sender, const std::string& inputID, const std::string& whatArg) :
LoggedError(sender, "ParsingError: during the parsing of " + inputID + " - " + whatArg)
{

}
