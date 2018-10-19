#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/order/order.hpp"

namespace mv
{

    namespace attr
    {

        static mv::json::Value toJSON(const Attribute& a)
        { 
            auto o = a.get<Order>();
            return json::Value(o.toString());
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::String)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to mv::Order");
            
            return Order(v.get<std::string>());
        }

        static std::string toString(const Attribute& a)
        {
            return a.get<Order>().toString();
        }

        MV_REGISTER_ATTR(Order)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setToStringFunc(toString);

    }

}
