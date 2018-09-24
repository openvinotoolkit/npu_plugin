#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{

    namespace attr
    {

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

        MV_REGISTER_ATTR(double)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setToStringFunc(toString);

    }

}