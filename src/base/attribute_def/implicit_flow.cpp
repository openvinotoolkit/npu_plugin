#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/computation/flow/implicit_flow.hpp"

namespace mv
{
    namespace attr_implicit_flow
    {
        static mv::json::Value toJSON(const Attribute& a)
        {
            auto implicitness = a.get<ImplicitFlow>();
            return json::Value( implicitness.isImplicit());
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if( v.valueType() != json::JSONType::Bool)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to bool");
            ImplicitFlow implicitness(v.get<bool>());

            return implicitness;
        }

        static std::string toString(const Attribute& a)
        {
            auto implicitness = a.get<ImplicitFlow>();
            return json::Value(implicitness.isImplicit()).stringify();
        }

        static std::vector<uint8_t> toBinary(const Attribute&a)
        {
            auto implicitness = a.get<ImplicitFlow>();
            return std::vector<uint8_t>(1,implicitness.isImplicit());
        }

    }

    namespace attr {
        MV_REGISTER_ATTR(ImplicitFlow)
            .setToJSONFunc(attr_implicit_flow::toJSON)
            .setFromJSONFunc(attr_implicit_flow::fromJSON)
            .setToStringFunc(attr_implicit_flow::toString)
            .setToBinaryFunc(attr_implicit_flow::toBinary);
    }


    
}
