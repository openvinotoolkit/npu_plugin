#ifndef MV_JSON_KEY_ERROR_HPP_
#define MV_JSON_KEY_ERROR_HPP_

#include <stdexcept>

namespace mv
{
    namespace json
    {

        class KeyError : public std::runtime_error
        {

        public:

            explicit KeyError(const std::string& whatArg);

        };

    }

}

#endif // MV_JSON_KEY_ERROR_HPP_