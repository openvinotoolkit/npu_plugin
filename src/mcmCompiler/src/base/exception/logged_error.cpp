#include "include/mcm/base/exception/logged_error.hpp"


mv::LoggedError::LoggedError(const LogSender& sender, const std::string& whatArg) :
std::runtime_error(sender.getLogID() + " - " + whatArg)
{
    sender.log(Logger::MessageType::Error, whatArg);
}

mv::LoggedError::LoggedError(const std::string& senderID, const std::string& whatArg) :
std::runtime_error(senderID + " - " + whatArg)
{
    Logger::instance().log(Logger::MessageType::Error, senderID, whatArg);
}


mv::LoggedError::~LoggedError()
{

}