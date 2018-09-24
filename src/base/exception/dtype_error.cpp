#include "include/mcm/base/exception/dtype_error.hpp"

mv::DTypeError::DTypeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "DTypeError: " + whatArg)
{

}