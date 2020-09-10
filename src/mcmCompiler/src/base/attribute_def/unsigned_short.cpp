#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_unsigned_short
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(static_cast<long long>(a.get<unsigned short>()));
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberInteger)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to unsigned short");
            return static_cast<unsigned short>(v.get<long long>());
        }

        static std::string toString(const Attribute& a)
        {
            return std::to_string(a.get<unsigned short>());
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            union Tmp
            {
                unsigned n;
                uint8_t bytes[sizeof(unsigned short)];
            };
            Tmp tmp = {a.get<unsigned short>()};
            return std::vector<uint8_t>(std::begin(tmp.bytes), std::end(tmp.bytes));
        }

    }

    namespace attr {
        MV_REGISTER_ATTR(unsigned short)
            .setToJSONFunc(attr_unsigned_short::toJSON)
            .setFromJSONFunc(attr_unsigned_short::fromJSON)
            .setToStringFunc(attr_unsigned_short::toString)
            .setToBinaryFunc(attr_unsigned_short::toBinary);


    }

}
