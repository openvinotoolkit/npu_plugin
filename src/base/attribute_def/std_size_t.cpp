#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_std_size_t
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(static_cast<long long>(a.get<std::size_t>()));
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberInteger)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to std::size_t");
            return static_cast<std::size_t>(v.get<long long>());
        }

        static std::string toString(const Attribute& a)
        {
            return std::to_string(a.get<std::size_t>());
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            union Tmp
            {
                std::size_t n;
                uint8_t bytes[sizeof(std::size_t)];
            };
            Tmp tmp = {a.get<std::size_t>()};
            return std::vector<uint8_t>(std::begin(tmp.bytes), std::end(tmp.bytes));
        }


    }

    namespace attr {
        MV_REGISTER_DUPLICATE_ATTR(std::size_t)
            .setToJSONFunc(attr_std_size_t::toJSON)
            .setFromJSONFunc(attr_std_size_t::fromJSON)
            .setToStringFunc(attr_std_size_t::toString)
            .setToBinaryFunc(attr_std_size_t::toBinary);
    }

}
