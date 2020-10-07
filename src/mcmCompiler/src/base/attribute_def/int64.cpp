#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_int64
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(static_cast<long long>(a.get<int64_t>()));
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberInteger)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to int");
            return static_cast<int64_t>(v.get<long long>());
        }

        static std::string toString(const Attribute& a)
        {
            return std::to_string(a.get<int64_t>());
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            union Tmp
            {
                int64_t n;
                uint8_t bytes[sizeof(int64_t)];
            };
            Tmp tmp = {a.get<int64_t>()};
            return std::vector<uint8_t>(std::begin(tmp.bytes), std::end(tmp.bytes));
        }

    }

    namespace attr {
        MV_REGISTER_ATTR(int64_t)
            .setToJSONFunc(attr_int64::toJSON)
            .setFromJSONFunc(attr_int64::fromJSON)
            .setToStringFunc(attr_int64::toString)
            .setToBinaryFunc(attr_int64::toBinary);

    }

}
