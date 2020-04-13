#ifndef MV_OP_ERROR_HPP_
#define MV_OP_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class OpError : public LoggedError
    {

    public:
            
        explicit OpError(const LogSender& sender, const std::string& whatArg);
        explicit OpError(const std::string& senderID, const std::string& whatArg);
        
    };

}

#endif // MV_OP_ERROR_HPP_