#ifndef MV_BINARYDATA_ERROR_HPP_
#define MV_BINARYDATA_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class BinaryDataError : public LoggedError
    {

    public:

        explicit BinaryDataError(const std::string& senderID, const std::string& whatArg);
    };

}

#endif // MV_BINARYDATA_ERROR_HPP_