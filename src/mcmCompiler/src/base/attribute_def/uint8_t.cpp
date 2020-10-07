#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_uint8_t
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(static_cast<long long>(a.get<uint8_t>()));
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberInteger)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                                        " to int");
            return static_cast<uint8_t>(v.get<long long>());
        }

        static std::string toString(const Attribute& a)
        {
            return std::to_string(a.get<uint8_t>());
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            return std::vector<uint8_t>(a.get<uint8_t>());
        }

    }

    namespace attr {

        MV_REGISTER_ATTR(uint8_t)
                .setToJSONFunc(attr_uint8_t::toJSON)
                .setFromJSONFunc(attr_uint8_t::fromJSON)
                .setToStringFunc(attr_uint8_t::toString)
                .setToBinaryFunc(attr_uint8_t::toBinary);

    }

}
