#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_int
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(static_cast<long long>(a.get<int>()));
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberInteger)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to int");
            return static_cast<int>(v.get<long long>());
        }

        static std::string toString(const Attribute& a)
        {
            return std::to_string(a.get<int>());
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            union Tmp
            {
                int n;
                uint8_t bytes[sizeof(int)];
            };
            Tmp tmp = {a.get<int>()};
            return std::vector<uint8_t>(std::begin(tmp.bytes), std::end(tmp.bytes));
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(int)
            .setToJSONFunc(attr_int::toJSON)
            .setFromJSONFunc(attr_int::fromJSON)
            .setToStringFunc(attr_int::toString)
            .setToBinaryFunc(attr_int::toBinary);
    }

}
