#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr
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

        MV_REGISTER_ATTR(bool)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setToStringFunc(toString)
            .setToBinaryFunc(toBinary);

    }

}
