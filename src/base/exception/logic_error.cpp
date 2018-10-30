#include "include/mcm/base/exception/logic_error.hpp"

mv::LogicError::LogicError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "LogicError: " + whatArg)
{

}