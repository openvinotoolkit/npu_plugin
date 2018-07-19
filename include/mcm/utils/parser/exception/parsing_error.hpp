#ifndef MV_PARSING_ERROR_HPP_
#define MV_PARSING_ERROR_HPP_

#include <stdexcept>

namespace mv
{

    class ParsingError : public std::runtime_error
    {

    public:

        explicit ParsingError(const std::string& whatArg);

    };

}

#endif // MV_PARSING_ERROR_HPP_