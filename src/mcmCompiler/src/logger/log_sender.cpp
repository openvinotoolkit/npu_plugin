#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/base/element.hpp"

mv::LogSender::~LogSender()
{

}

void mv::LogSender::log(Logger::MessageType messageType, const std::string &content) const
{
    #if MV_LOG_ENABLED
        Logger::log(messageType, getLogID(), content);
    #endif
}