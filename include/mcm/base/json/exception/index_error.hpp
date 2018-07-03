#ifndef MV_JSON_INDEX_ERROR_HPP_
#define MV_JSON_INDEX_ERROR_HPP_

#include <stdexcept>

namespace mv
{
    namespace json
    {

        class IndexError : public std::runtime_error
        {

        public:

            explicit IndexError(const std::string& whatArg);

        };

    }

}

#endif // MV_JSON_INDEX_ERROR_HPP_