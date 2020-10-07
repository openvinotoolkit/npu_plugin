#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr_bool
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return json::Value(a.get<bool>());
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Bool)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to bool");
            return v.get<bool>();
        }

        static std::string toString(const Attribute& a)
        {
            return json::Value(a.get<bool>()).stringify();
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            return std::vector<uint8_t>(1, a.get<bool>());
        }
    }

    namespace attr {
        MV_REGISTER_ATTR(bool)
            .setToJSONFunc(attr_bool::toJSON)
            .setFromJSONFunc(attr_bool::fromJSON)
            .setToStringFunc(attr_bool::toString)
            .setToBinaryFunc(attr_bool::toBinary);
   }
}
