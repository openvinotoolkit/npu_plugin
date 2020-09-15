#ifndef LOG_SENDER_HPP_
#define LOG_SENDER_HPP_

#include <string>
#include "include/mcm/logger/logger.hpp"

namespace mv
{
    
    class Element;
    
    class LogSender
    {    

    public:

        virtual ~LogSender() = 0;
        virtual std::string getLogID() const = 0;
        void log(Logger::MessageType messageType, const std::string &content) const;
        
    };

}

#endif // LOG_SENDER_HPP_