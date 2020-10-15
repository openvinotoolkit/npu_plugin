#ifndef MV_ARGUMENT_ERROR_HPP_
#define MV_ARGUMENT_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class ArgumentError : public LoggedError
    {

        std::string argName_;
        std::string argVal_;

    public:

        explicit ArgumentError(const LogSender& sender, const std::string& argName,
            const std::string& argVal, const std::string& whatArg);
        explicit ArgumentError(const std::string& senderID, const std::string& argName,
            const std::string& argVal, const std::string& whatArg);

    };

}

#endif // MV_ARGUMENT_ERROR_HPP_
