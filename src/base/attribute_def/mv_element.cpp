#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/element.hpp"

namespace mv
{

    namespace attr
    {

        static mv::json::Value toJSON(const Attribute& a)
        { 
            return a.get<Element>().toJSON();
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Object)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to mv::Element");
            return Element(v.get<json::Object>());
        }

        static Attribute fromSimplifiedJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Object)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to mv::Element");
            return Element(v.get<json::Object>(), true);
        }

        static std::string toString(const Attribute& a)
        {
            return a.get<Element>().toString();
        }

        MV_REGISTER_ATTR(Element)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setFromSimplifiedJSONFunc(fromSimplifiedJSON)
            .setToStringFunc(toString);

    }

}