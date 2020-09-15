#ifndef MV_LOGGED_ERROR_HPP_
#define MV_LOGGED_ERROR_HPP_

#include <stdexcept>
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{



    class LoggedError : public std::runtime_error
    {

    public:

        explicit LoggedError(const LogSender& sender, const std::string& whatArg);
        explicit LoggedError(const std::string& senderID, const std::string& whatArg);
        virtual ~LoggedError() = 0;

    };

}

#endif // MV_LOGGED_ERROR_HPP_