#ifndef err1
#define err1

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class err1C : public LoggedError
    {

    public:
            
        explicit err1C(const LogSender& sender, const std::string& whatArg);
        
    };

}

#endif // err1