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