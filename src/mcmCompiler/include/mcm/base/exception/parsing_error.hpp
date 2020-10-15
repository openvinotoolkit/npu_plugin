#ifndef MV_PARSING_ERROR_HPP_
#define MV_PARSING_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class ParsingError : public LoggedError
    {

    public:

        explicit ParsingError(const LogSender& sender, const std::string& inputID, const std::string& whatArg);

    };

}

#endif // MV_PARSING_ERROR_HPP_