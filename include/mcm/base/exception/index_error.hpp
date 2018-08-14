#ifndef MV_INDEX_ERROR_HPP_
#define MV_INDEX_ERROR_HPP_

#include <stdexcept>

namespace mv
{

    class IndexError : public std::runtime_error
    {

    public:

        explicit IndexError(const std::string& whatArg);

    };

}

#endif // MV_INDEX_ERROR_HPP_