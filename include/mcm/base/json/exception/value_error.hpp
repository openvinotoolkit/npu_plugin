#ifndef MV_VALUE_KEY_ERROR_HPP_
#define MV_VALUE_KEY_ERROR_HPP_

#include <stdexcept>

namespace mv
{
    namespace json
    {

        class ValueError : public std::runtime_error
        {

        public:

            explicit ValueError(const std::string& whatArg);

        };

    }

}

#endif // MV_VALUE_KEY_ERROR_HPP_