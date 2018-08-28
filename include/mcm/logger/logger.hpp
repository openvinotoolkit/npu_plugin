#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <time.h>
#include <string>
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/base/printable.hpp"

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
        
        std::string getTime() const;
        void logMessage(MessageType messageType, std::string content) const;

    protected:

        virtual void logError(const std::string &content) const = 0;
        virtual void logWarning(const std::string &content) const = 0;
        virtual void logInfo(const std::string &content) const = 0;
        virtual void logDebug(const std::string &content) const = 0;

    public:

        Logger(VerboseLevel verboseLevel, bool logTime);
        virtual ~Logger() = 0;
        void setVerboseLevel(VerboseLevel verboseLevel);
        void setLogTime(bool logTime);
        void log(MessageType messageType, const std::string &content) const;

    };

}

#endif // LOGGER_HPP_