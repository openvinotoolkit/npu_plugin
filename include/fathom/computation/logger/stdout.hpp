#ifndef STDOUT_HPP_
#define STDOUT_HPP_

#include <iostream>
#include "include/fathom/computation/logger/logger.hpp"

namespace mv
{

    class StdOutLogger : public Logger
    {

        void logError(const string &content) const
        {
            std::cerr << content << std::endl;
        }

        void logWarning(const string &content) const
        {
            std::cerr << content << std::endl;
        }

        void logInfo(const string &content) const
        {
            std::cout << content << std::endl;
        }

    public:

        StdOutLogger(VerboseLevel verboseLevel = VerboseSilent, bool outputTime = true) :
        Logger(verboseLevel, outputTime)
        {
            
        }

    };

}

#endif // STDOUT_HPP_