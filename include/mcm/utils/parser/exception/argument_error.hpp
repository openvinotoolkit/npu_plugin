#ifndef MV_ARGUMENT_ERROR_HPP_
#define MV_ARGUMENT_ERROR_HPP_

#include <stdexcept>

namespace mv
{

    class ArgumentError : public std::runtime_error
    {

    public:

        explicit ArgumentError(const std::string& whatArg);

    };

}

#endif // MV_ARGUMENT_ERROR_HPP_