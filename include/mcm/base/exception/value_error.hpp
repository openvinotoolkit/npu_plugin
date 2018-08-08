#ifndef MV_VALUE_ERROR_HPP_
#define MV_VALUE_ERROR_HPP_

#include <stdexcept>

namespace mv
{

    class ValueError : public std::runtime_error
    {

    public:

        explicit ValueError(const std::string& whatArg);

    };

}

#endif // MV_VALUE_ERROR_HPP_