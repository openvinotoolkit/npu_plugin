#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#define printfInfo(senderName, ...) mv::Logger::DebugLog(senderName, __VA_ARGS__)

#include <time.h>
#include <string>
#include <unordered_map>
#include <stdarg.h>
#include <stdio.h>
#include <iostream>
#include <regex>
#include <list>

namespace mv
{

    enum class VerboseLevel
    {
        Silent,
        Error,
        Warning,
        Info,
        Debug
    };

    class Logger
    {

    public:

        enum class MessageType
        {
            Debug,
            Info,
            Warning,
            Error
        };

    private:

        VerboseLevel verboseLevel_;
        bool logTime_;
        std::string indent_;
        std::list<std::regex> filterList_;
        bool filterPositive_;
        
        std::string getTime_() const;
        void logMessage_(MessageType messageType, std::string content) const;

        void logError_(const std::string &content) const;
        void logWarning_(const std::string &content) const;
        void logInfo_(const std::string &content) const;
        void logDebug_(const std::string &content) const;

        Logger();
        Logger(const Logger& other) = delete; 
        Logger& operator=(const Logger& other) = delete;
        ~Logger();

    public:

        static Logger& instance();
        static void DebugLog(const std::string& senderName, const char* const format, ...);
        static void log(MessageType messageType, const std::string& senderName, const std::string &content);
        static void setVerboseLevel(VerboseLevel verboseLevel);
        static VerboseLevel getVerboseLevel();
        static void enableLogTime();
        static void disableLogTime();
        static void logFilter(std::list<std::regex> filterList, bool filterPositive);
        static void clearFilter();

    };

    inline bool isDebugFilesEnabled() {
        const auto level = Logger::getVerboseLevel();
        return (VerboseLevel::Error != level && VerboseLevel::Silent != level);
    }
}

#endif // LOGGER_HPP_
