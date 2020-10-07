#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>

namespace mv
{
    namespace attr_std_vector_float
    {
        static std::string toString(const Attribute& a)
        {
            std::string output = "(" + std::to_string(a.get<std::vector<float>>().size()) + ")";
            return output;
        }

        static std::string toLongString(const Attribute& a)
        {
            std::string output = "{";
            auto vec = a.get<std::vector<float>>();
            if (vec.size() > 0)
            {
                for (std::size_t i = 0; i < vec.size() - 1; ++i)
                    output += std::to_string(vec[i]) + ", ";
                output += std::to_string(vec.back());
            }
            return output + "}";
        }
    }

    namespace attr {
        // NB: This attribute isn't used by JSON reader,
        // so it isn't necessary to implement corresponding functions here
        MV_REGISTER_ATTR(std::vector<float>)
            .setToStringFunc(attr_std_vector_float::toString)
            .setToStringFunc(attr_std_vector_float::toLongString)
            .setTypeTrait("large");
    }
}
