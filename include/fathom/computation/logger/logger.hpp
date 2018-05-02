#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <time.h>
#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    class Logger
    {

    public:

        enum VerboseLevel
        {
            VerboseSilent,
            VerboseWarning,
            VerboseError,
            VerboseDebug
        };

        enum MessageType
        {
            MessageInfo,
            MessageWarning,
            MessageError
        };

    private:

        VerboseLevel verboseLevel_;
        bool logTime_;
        
        string getTime() const;
        void logMessage(MessageType messageType, const string &content) const;

    protected:

        virtual void logError(const string &content) const = 0;
        virtual void logWarning(const string &content) const = 0;
        virtual void logInfo(const string &content) const = 0;

    public:

        Logger(VerboseLevel verboseLevel, bool logTime);
        virtual ~Logger() = 0;
        void setVerboseLevel(VerboseLevel verboseLevel);
        void log(MessageType messageType, const string &content) const;

    };

}

#endif // LOGGER_HPP_