#ifndef MV_INDEX_ERROR_HPP_
#define MV_INDEX_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class IndexError : public LoggedError
    {

    public:

        explicit IndexError(const LogSender& sender, long long idx, const std::string& whatArg);
        explicit IndexError(const std::string& senderID, long long idx, const std::string& whatArg);

    };

}

#endif // MV_INDEX_ERROR_HPP_