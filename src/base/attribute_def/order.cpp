#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/order/order.hpp"

namespace mv
{

    namespace attr_order
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
                    " to Order");
            
            return Order(v.get<std::string>());
        }

        static std::string toString(const Attribute& a)
        {
            return "mv::Order(\"" + a.get<Order>().toString() + "\")";
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(Order)
            .setToJSONFunc(attr_order::toJSON)
            .setFromJSONFunc(attr_order::fromJSON)
            .setToStringFunc(attr_order::toString);
    }

}
