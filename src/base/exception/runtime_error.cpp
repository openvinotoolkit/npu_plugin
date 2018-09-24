#include "include/mcm/base/exception/runtime_error.hpp"

mv::RuntimeError::RuntimeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "RuntimeError: " + whatArg)
{

}