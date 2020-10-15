#include "include/mcm/base/exception/order_error.hpp"

mv::OrderError::OrderError(const LogSender& sender, const std::string& whatArg)
    :LoggedError(sender, "OrderError: " + whatArg)
{

}

mv::OrderError::OrderError(const std::string& senderID, const std::string& whatArg)
    : LoggedError(senderID, "OrderError: " + whatArg)
{

}
