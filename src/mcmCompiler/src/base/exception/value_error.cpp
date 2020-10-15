#include "include/mcm/base/exception/value_error.hpp"

mv::ValueError::ValueError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "ValueError: " + whatArg)
{

}