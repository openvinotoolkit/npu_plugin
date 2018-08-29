#ifndef LOG_SENDER_HPP_
#define LOG_SENDER_HPP_

#include <string>
#include <include/mcm/logger/logger.hpp>

namespace mv
{

    class LogSender
    {    
        
    protected:

        virtual std::string getLogID_() const = 0;

    public:

        virtual ~LogSender();
        void log(Logger::MessageType messageType, const std::string &content) const;

    };

}

#endif // LOG_SENDER_HPP_