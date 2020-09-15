#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"

namespace mv
{

    namespace attr_dtype
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            auto d = a.get<DType>();
            return json::Value(d.toString());
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::String)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to mv::DType");

            return DType(v.get<std::string>());
        }

        static std::string toString(const Attribute& a)
        {
            return "mv::DType(\""+a.get<DType>().toString()+"\")";
        }
    }


    namespace attr {
        MV_REGISTER_ATTR(DType)
            .setToJSONFunc(attr_dtype::toJSON)
            .setFromJSONFunc(attr_dtype::fromJSON)
            .setToStringFunc(attr_dtype::toString);
    }

}
