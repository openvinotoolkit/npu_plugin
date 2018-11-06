#ifndef MV_ORDER_ERROR_HPP_
#define MV_ORDER_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class OrderError : public LoggedError
    {

    public:
            
        explicit OrderError(const LogSender& sender, const std::string& whatArg);
        
    };

}

#endif //MV_ORDER_ERROR_HPP_
