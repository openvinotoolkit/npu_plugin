#include "include/fathom/computation/logger/logger.hpp"

mv::string mv::Logger::getTime() const
{
    struct tm *timeInfo;
    time_t rawTime;
    char buffer[80];
    time(&rawTime);
    timeInfo = localtime(&rawTime);
    strftime(buffer, 80, "%T %D", timeInfo);
    return string(buffer);
}

void mv::Logger::logMessage(MessageType messageType, const string &content) const
{

    string logMessage;

    if (logTime_)
        logMessage += getTime() + "\t";

    switch (messageType)
    {
        case MessageError:
            logMessage += "ERROR:\t";
            logMessage += content;
            logError(logMessage);
            break;

        case MessageWarning:
            logMessage += "WARNING:\t";
            logMessage += content;
            logWarning(logMessage);
            break;

        default:
            logMessage += "INFO:\t";
            logMessage += content;
            logInfo(logMessage);
            break;

    }

}

mv::Logger::Logger(VerboseLevel verboseLevel, bool logTime) : 
verboseLevel_(verboseLevel), 
logTime_(logTime)
{

}

mv::Logger::~Logger()
{
    
}

void mv::Logger::setVerboseLevel(VerboseLevel verboseLevel)
{
    verboseLevel_ = verboseLevel;
}

void mv::Logger::log(MessageType messageType, const string &content) const
{

    switch (verboseLevel_)
    {

        case VerboseDebug:
            logMessage(messageType, content);
            break;

        case VerboseWarning:
            if (messageType == MessageError || messageType == MessageWarning)
                logMessage(messageType, content);
            break;

        case VerboseError:
            if (messageType == MessageError)
                logMessage(messageType, content);
            break;

        default:
            break;

    }

}