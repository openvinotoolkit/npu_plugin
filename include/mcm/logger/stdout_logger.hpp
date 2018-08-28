#ifndef STDOUT_HPP_
#define STDOUT_HPP_

#include <iostream>
#include "include/mcm/logger/logger.hpp"

namespace mv
{

    class StdOutLogger : public Logger
    {

        void logError(const std::string &content) const;
        void logWarning(const std::string &content) const;
        void logInfo(const std::string &content) const;
        void logDebug(const std::string &content) const;

    public:

        StdOutLogger(VerboseLevel verboseLevel = VerboseLevel::VerboseSilent, bool outputTime = true);
    };

}

#endif // STDOUT_HPP_