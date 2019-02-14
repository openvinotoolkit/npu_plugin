#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

namespace mv
{

    namespace attr
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            auto q = a.get<mv::QuantizationParams>();
            return q.toJSON();
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Object)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to mv::QuantizationParams");
            return mv::QuantizationParams(v.get<json::Object>());
        }

        static std::string toString(const Attribute& a)
        {
            return "QuantizationParams::" + a.get<QuantizationParams>().toString();
        }

        MV_REGISTER_ATTR(QuantizationParams)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setToStringFunc(toString);

    }

}
