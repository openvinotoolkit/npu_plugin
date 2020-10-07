#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_unsigned
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(static_cast<long long>(a.get<unsigned>()));
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberInteger)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to unsigned");
            return static_cast<unsigned>(v.get<long long>());
        }

        static std::string toString(const Attribute& a)
        {
            return std::to_string(a.get<unsigned>());
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            union Tmp
            {
                unsigned n;
                uint8_t bytes[sizeof(unsigned)];
            };
            Tmp tmp = {a.get<unsigned>()};
            return std::vector<uint8_t>(std::begin(tmp.bytes), std::end(tmp.bytes));
        }


    }


    namespace attr {

        MV_REGISTER_DUPLICATE_ATTR(unsigned)
            .setToJSONFunc(attr_unsigned::toJSON)
            .setFromJSONFunc(attr_unsigned::fromJSON)
            .setToStringFunc(attr_unsigned::toString)
            .setToBinaryFunc(attr_unsigned::toBinary);


    }

}
