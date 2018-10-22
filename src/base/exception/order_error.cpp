#include "include/mcm/base/exception/order_error.hpp"

mv::OrderError::OrderError(const LogSender& sender, const std::string& whatArg)
    :LoggedError(sender, "mv::OrderError: " + whatArg)
{

}
