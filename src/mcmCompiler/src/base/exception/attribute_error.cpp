#include "include/mcm/base/exception/attribute_error.hpp"

mv::AttributeError::AttributeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "AttributeError: " + whatArg)
{

}

mv::AttributeError::AttributeError(const std::string& senderID, const std::string& whatArg) :
LoggedError(senderID, "AttributeError: " + whatArg)
{

}