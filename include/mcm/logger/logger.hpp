#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <time.h>
#include <string>
#include <unordered_map>
#include <iostream>

namespace mv
{

    class Logger
    {

    public:

        enum class VerboseLevel
        {
            VerboseSilent,
            VerboseError,
            VerboseWarning,
            VerboseInfo,
            VerboseDebug
        };

        enum class MessageType
        {
            MessageDebug,
            MessageInfo,
            MessageWarning,
            MessageError
        };

    private:

        VerboseLevel verboseLevel_;
        bool logTime_;
        std::string indent_;
        
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
        static void log(MessageType messageType, const std::string& senderName, const std::string &content);
        static void setVerboseLevel(VerboseLevel verboseLevel);
        static void enableLogTime();
        static void disableLogTime();

    };

}

#endif // LOGGER_HPP_