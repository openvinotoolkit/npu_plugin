#include "include/mcm/logger/logger.hpp"
#include "include/mcm/base/printable.hpp"

void mv::Logger::DebugLog(const std::string& senderName, char* format, ...) {
    va_list args;
    va_start(args, format);
    char buffer[256];
    std::string logString;
    vsnprintf(buffer, 256, format, args);
    logString = buffer;
    mv::Logger::log(mv::Logger::MessageType::Debug, senderName, logString);
    va_end(args);
}

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
        case MessageType::Error:
            logContent += "ERROR:   ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logError_(logContent);
            break;

        case MessageType::Warning:
            logContent += "WARNING: ";
            Printable::replaceSub(content, "\n", "\n" + indent_ + "         ");
            Printable::replaceSub(content, "\n\t", "\n" + indent_ + "            ");
            logContent += content;
            logWarning_(logContent);
            break;

        case MessageType::Info:
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
verboseLevel_(VerboseLevel::Error), 
logTime_(false),
filterPositive_(true)
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

mv::VerboseLevel mv::Logger::getVerboseLevel()
{
    return instance().verboseLevel_;
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
    #if MV_LOG_ENABLED
    if (!instance().filterList_.empty())
    {
        bool found = false;
        for (auto it = instance().filterList_.begin(); it != instance().filterList_.end(); ++it)
        {
            std::smatch m;
            if (std::regex_match(senderName, m, *it))
            {
                if (!instance().filterPositive_)
                    return;
                else
                {
                    found = true;
                    break;
                }
            }
        }

        if (!found && instance().filterPositive_)
            return;

    }

    switch (instance().verboseLevel_)
    {

         case VerboseLevel::Debug:
            instance().logMessage_(messageType, senderName + " - " + content);
            break;

        case VerboseLevel::Info:
            if (messageType == MessageType::Error || messageType == MessageType::Warning || messageType == MessageType::Info)
                instance().logMessage_(messageType, senderName + " - " + content);
            break;

        case VerboseLevel::Warning:
            if (messageType == MessageType::Error || messageType == MessageType::Warning)
                instance().logMessage_(messageType, senderName + " - " + content);
            break;

        case VerboseLevel::Error:
            if (messageType == MessageType::Error)
                instance().logMessage_(messageType, senderName + " - " + content);
            break;

        default:
            break;

    }
    #endif
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

void mv::Logger::logFilter(std::list<std::regex> filterList, bool filterPositive)
{
    instance().filterList_ = filterList;
    instance().filterPositive_ = filterPositive;
}

void mv::Logger::clearFilter()
{
    instance().filterList_.clear();
    instance().filterPositive_ = true;
}
