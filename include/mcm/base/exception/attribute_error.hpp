#ifndef MV_ATTRIBUTE_ERROR_HPP_
#define MV_ATTRIBUTE_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class AttributeError : public LoggedError
    {

    public:

        explicit AttributeError(const LogSender& sender, const std::string& whatArg);

    };

}

#endif // MV_ATTRIBUTE_ERROR_HPP_