#include "include/fathom/computation/logger/logger.hpp"

mv::string mv::Logger::getTime() const
{
    struct tm *timeInfo;
    time_t rawTime;
    char buffer[18];
    time(&rawTime);
    timeInfo = localtime(&rawTime);
    strftime(buffer, 18, "%T %D", timeInfo);
    return string(buffer);
}

void mv::Logger::logMessage(MessageType messageType, string content) const
{

    string logContent;

    if (logTime_)
        logContent += getTime() + " ";

    switch (messageType)
    {
        case MessageType::MessageError:
            logContent += "ERROR:   ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logError(logContent);
            break;

        case MessageType::MessageWarning:
            logContent += "WARNING: ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logWarning(logContent);
            break;

        case MessageType::MessageInfo:
            logContent += "INFO:    ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logInfo(logContent);
            break;

        default:
            logContent += "DEBUG:   ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logDebug(logContent);
            break;

    }

}

mv::Logger::Logger(VerboseLevel verboseLevel, bool logTime) : 
verboseLevel_(verboseLevel), 
logTime_(logTime)
{
    if (logTime_)
        indent_ = "                     ";
    else
        indent_ = "   ";
}

mv::Logger::~Logger()
{
    
}

void mv::Logger::setVerboseLevel(VerboseLevel verboseLevel)
{
    verboseLevel_ = verboseLevel;
}

void mv::Logger::setLogTime(bool logTime)
{
    logTime_ = logTime;
    if (logTime_)
        indent_ = "                     ";
    else
        indent_ = "   ";
}

void mv::Logger::log(MessageType messageType, const string &content) const
{

    switch (verboseLevel_)
    {

         case VerboseLevel::VerboseDebug:
            logMessage(messageType, content);
            break;

        case VerboseLevel::VerboseInfo:
            if (messageType == MessageType::MessageError || messageType == MessageType::MessageWarning || messageType == MessageType::MessageInfo)
                logMessage(messageType, content);
            break;

        case VerboseLevel::VerboseWarning:
            if (messageType == MessageType::MessageError || messageType == MessageType::MessageWarning)
                logMessage(messageType, content);
            break;

        case VerboseLevel::VerboseError:
            if (messageType == MessageType::MessageError)
                logMessage(messageType, content);
            break;

        default:
            break;

    }

}