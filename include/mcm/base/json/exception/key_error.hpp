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