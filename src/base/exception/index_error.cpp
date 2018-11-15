#include "include/mcm/base/exception/index_error.hpp"

mv::IndexError::IndexError(const LogSender& sender, long long idx, const std::string& whatArg) :
LoggedError(sender, "IndexError: index " + std::to_string(idx) + " - " + whatArg)
{

}

mv::IndexError::IndexError(const std::string& senderID, long long idx, const std::string& whatArg) :
LoggedError(senderID, "IndexError: index " + std::to_string(idx) + " - " + whatArg)
{

}