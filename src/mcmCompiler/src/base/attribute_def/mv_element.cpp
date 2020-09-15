#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/element.hpp"

namespace mv
{

    namespace attr_mv_element
    {

        static mv::json::Value toJSON(const Attribute& a)
        { 
            return a.get<Element>().toJSON();
        }

        static mv::json::Value toSimplifiedJSON(const Attribute& a)
        {
            return a.get<Element>().toJSON(true);
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

    }

    namespace attr {
        MV_REGISTER_ATTR(Element)
            .setToJSONFunc(attr_mv_element::toJSON)
            .setFromJSONFunc(attr_mv_element::fromJSON)
            .setToSimplifiedJSONFunc(attr_mv_element::toSimplifiedJSON)
            .setFromSimplifiedJSONFunc(attr_mv_element::fromSimplifiedJSON)
            .setToStringFunc(attr_mv_element::toString);

    }

}
