#include "include/mcm/base/exception/master_error.hpp"

mv::MasterError::MasterError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "*MasterError* - " + whatArg + " - aborting execution")
{
    exit(1);
}