#ifndef MV_DTYPE_ERROR_HPP_
#define MV_DTYPE_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class DTypeError : public LoggedError
    {

    public:

        explicit DTypeError(const LogSender& sender, const std::string& whatArg);
        explicit DTypeError(const std::string& senderID, const std::string& whatArg);

    };

}

#endif // MV_DTYPE_ERROR_HPP_