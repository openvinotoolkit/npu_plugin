#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_double
    {
        template <typename T>
        static std::string toStringWithPrecision(const T val, const int precision = 15)
        {
            std::ostringstream outstream;
            outstream.precision(precision);
            outstream << std::fixed << val;
            return outstream.str();
        }

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(a.get<double>());
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberFloat)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to double");
            return v.get<double>();
        }

        static std::string toString(const Attribute& a)
        {
            return std::to_string(a.get<double>());
        }

        static std::string toLongString(const Attribute& a)
        {
            return toStringWithPrecision(a.get<double>());
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            union Tmp
            {
                double n;
                uint8_t bytes[sizeof(double)];
            };
            Tmp tmp = {a.get<double>()};
            return std::vector<uint8_t>(std::begin(tmp.bytes), std::end(tmp.bytes));
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(double)
            .setToJSONFunc(attr_double::toJSON)
            .setFromJSONFunc(attr_double::fromJSON)
            .setToStringFunc(attr_double::toString)
            .setToLongStringFunc(attr_double::toLongString)
            .setToBinaryFunc(attr_double::toBinary)
            .setTypeTrait("large");
    }

}
