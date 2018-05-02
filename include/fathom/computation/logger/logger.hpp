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
        
        string getTime() const
        {
            struct tm *timeInfo;
            time_t rawTime;
            char buffer[80];
            time(&rawTime);
            timeInfo = localtime(&rawTime);
            strftime(buffer, 80, "%T %D", timeInfo);
            return string(buffer);
        }

        void logMessage(MessageType messageType, const string &content) const
        {

            string logMessage;

            if (logTime_)
                logMessage += getTime() + " ";

            switch (messageType)
            {
                case MessageError:
                    logMessage += "ERROR: ";
                    logMessage += content;
                    logError(logMessage);
                    break;

                case MessageWarning:
                    logMessage += "WARNING: ";
                    logMessage += content;
                    logWarning(logMessage);
                    break;

                default:
                    logMessage += "INFO: ";
                    logMessage += content;
                    logInfo(logMessage);
                    break;

            }

        }

    protected:

        virtual void logError(const string &content) const = 0;
        virtual void logWarning(const string &content) const = 0;
        virtual void logInfo(const string &content) const = 0;

    public:

        Logger(VerboseLevel verboseLevel, bool logTime) : verboseLevel_(verboseLevel), logTime_(logTime)
        {

        }

        void setVerboseLevel(VerboseLevel verboseLevel)
        {
            verboseLevel_ = verboseLevel;
        }

        void log(MessageType messageType, const string &content) const
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

    };

}

#endif // LOGGER_HPP_