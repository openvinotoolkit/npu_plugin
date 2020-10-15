#ifndef MV_SHAPE_ERROR_HPP_
#define MV_SHAPE_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class ShapeError : public LoggedError
    {

    public:

        explicit ShapeError(const LogSender& sender, const std::string& whatArg);
        
    };

}

#endif // MV_SHAPE_ERROR_HPP_