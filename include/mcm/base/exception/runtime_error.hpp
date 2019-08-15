#ifndef MV_RUNTIME_ERROR_HPP_
#define MV_RUNTIME_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class RuntimeError : public LoggedError
    {

    public:
            
        explicit RuntimeError(const LogSender& sender, const std::string& whatArg);
        explicit RuntimeError(const std::string& senderID, const std::string& whatArg);
        
    };

}

#endif // MV_RUNTIME_ERROR_HPP_