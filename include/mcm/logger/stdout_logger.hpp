#ifndef STDOUT_HPP_
#define STDOUT_HPP_

#include <iostream>
#include "include/mcm/logger/logger.hpp"

namespace mv
{

    class StdOutLogger : public Logger
    {

        void logError(const string &content) const;
        void logWarning(const string &content) const;
        void logInfo(const string &content) const;
        void logDebug(const string &content) const;

    public:

        StdOutLogger(VerboseLevel verboseLevel = VerboseLevel::VerboseSilent, bool outputTime = true);
    };

}

#endif // STDOUT_HPP_