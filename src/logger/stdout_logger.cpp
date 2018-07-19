#include "include/mcm/logger/stdout_logger.hpp"

mv::StdOutLogger::StdOutLogger(VerboseLevel verboseLevel, bool outputTime) :
Logger(verboseLevel, outputTime)
{
    
}

void mv::StdOutLogger::logError(const string &content) const
{
    std::cerr << content << std::endl;
}

void mv::StdOutLogger::logWarning(const string &content) const
{
    std::cerr << content << std::endl;
}

void mv::StdOutLogger::logInfo(const string &content) const
{
    std::cout << content << std::endl;
}

void mv::StdOutLogger::logDebug(const string &content) const
{
    std::cout << content << std::endl;
}


