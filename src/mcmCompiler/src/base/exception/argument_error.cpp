#include "include/mcm/base/exception/argument_error.hpp"

mv::ArgumentError::ArgumentError(const LogSender& sender, const std::string& argName, const std::string& argVal,
    const std::string& whatArg) :
LoggedError(sender, "ArgumentError: " + argName + " " + argVal + " - " + whatArg)
{

}

mv::ArgumentError::ArgumentError(const std::string& senderID, const std::string& argName, const std::string& argVal,
    const std::string& whatArg) :
LoggedError(senderID, "ArgumentError: " + argName + " " + argVal + " - " + whatArg)
{

}
