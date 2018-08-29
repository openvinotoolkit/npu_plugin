#include "include/mcm/logger/logger.hpp"

std::string mv::Logger::getTime_() const
{
    struct tm *timeInfo;
    time_t rawTime;
    char buffer[18];
    time(&rawTime);
    timeInfo = localtime(&rawTime);
    strftime(buffer, 18, "%T %D", timeInfo);
    return std::string(buffer);
}

void mv::Logger::logMessage_(MessageType messageType, std::string content) const
{

    std::string logContent;

    if (logTime_)
        logContent += getTime_() + " ";

    switch (messageType)
    {
        case MessageType::MessageError:
            logContent += "ERROR:   ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logError_(logContent);
            break;

        case MessageType::MessageWarning:
            logContent += "WARNING: ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logWarning_(logContent);
            break;

        case MessageType::MessageInfo:
            logContent += "INFO:    ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logInfo_(logContent);
            break;

        default:
            logContent += "DEBUG:   ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logDebug_(logContent);
            break;

    }

}

mv::Logger& mv::Logger::instance()
{
    static Logger instance_;
    return instance_;
}

mv::Logger::Logger() : 
verboseLevel_(VerboseLevel::VerboseInfo), 
logTime_(false)
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
    instance().verboseLevel_ = verboseLevel;
}

void mv::Logger::enableLogTime()
{
    instance().logTime_ = true;
    instance().indent_ = "                     ";
}

void mv::Logger::disableLogTime()
{
    instance().logTime_ = false;
    instance().indent_ = "   ";
}

void mv::Logger::log(MessageType messageType, const std::string& senderName, const std::string &content)
{

    switch (instance().verboseLevel_)
    {

         case VerboseLevel::VerboseDebug:
            instance().logMessage_(messageType, senderName + " - " + content);
            break;

        case VerboseLevel::VerboseInfo:
            if (messageType == MessageType::MessageError || messageType == MessageType::MessageWarning || messageType == MessageType::MessageInfo)
                instance().logMessage_(messageType, senderName + " - " + content);
            break;

        case VerboseLevel::VerboseWarning:
            if (messageType == MessageType::MessageError || messageType == MessageType::MessageWarning)
                instance().logMessage_(messageType, senderName + " - " + content);
            break;

        case VerboseLevel::VerboseError:
            if (messageType == MessageType::MessageError)
                instance().logMessage_(messageType, senderName + " - " + content);
            break;

        default:
            break;

    }

}

void mv::Logger::logError_(const std::string &content) const
{
    std::cerr << content << std::endl;
}

void mv::Logger::logWarning_(const std::string &content) const
{
    std::cerr << content << std::endl;
}

void mv::Logger::logInfo_(const std::string &content) const
{
    std::cout << content << std::endl;
}

void mv::Logger::logDebug_(const std::string &content) const
{
    std::cout << content << std::endl;
}
