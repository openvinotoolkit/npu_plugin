#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <string>

namespace mv
{

    namespace attr_std_string
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            return mv::json::Value(a.get<std::string>());
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::String)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to std::string");
            return v.get<std::string>();
        }

        static std::string toString(const Attribute& a)
        {
            return "\"" + a.get<std::string>() + "\"";
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(std::string)
            .setToJSONFunc(attr_std_string::toJSON)
            .setFromJSONFunc(attr_std_string::fromJSON)
            .setToStringFunc(attr_std_string::toString);
    }

}
