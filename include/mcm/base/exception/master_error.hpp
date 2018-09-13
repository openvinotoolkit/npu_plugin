#ifndef MV_MASTER_ERROR_HPP_
#define MV_MASTER_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class MasterError : public LoggedError
    {

    public:

        explicit MasterError(const LogSender& sender, const std::string& whatArg);

    };

}
#endif // MV_ARGUMENT_ERROR_HPP_