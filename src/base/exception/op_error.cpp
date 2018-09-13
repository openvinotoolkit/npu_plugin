#include "include/mcm/base/exception/op_error.hpp"

mv::OpError::OpError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "OpError: " + whatArg)
{

}