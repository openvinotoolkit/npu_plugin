#ifndef MV_LOGIC_ERROR_HPP_
#define MV_LOGIC_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class LogicError : public LoggedError
    {

    public:
            
        explicit LogicError(const LogSender& sender, const std::string& whatArg);
        
    };

}

#endif // MV_RUNTIME_ERROR_HPP_