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
            std::string output = "(" + std::to_string(a.get<mv::QuantizationParams>().getZeroPoint().size()) + ", " +
                    std::to_string(a.get<mv::QuantizationParams>().getScale().size()) + ", " +
                    std::to_string(a.get<mv::QuantizationParams>().getMin().size()) + ", " +
                    std::to_string(a.get<mv::QuantizationParams>().getMax().size()) + ")";
            return  output;
        }

        static std::string toLongString(const Attribute& a)
        {
            std::string output = "{{";
            auto vec1 = a.get<mv::QuantizationParams>().getZeroPoint();
            if (vec1.size() > 0)
            {
                for (std::size_t i = 0; i < vec1.size() - 1; ++i)
                    output += std::to_string(vec1[i]) + ", ";
                output += std::to_string(vec1.back());
            }
            output += "},{";
            auto vec2 = a.get<mv::QuantizationParams>().getScale();
            if (vec2.size() > 0)
            {
                for (std::size_t i = 0; i < vec2.size() - 1; ++i)
                    output += std::to_string(vec2[i]) + ", ";
                output += std::to_string(vec2.back());
            }
            output += "},{";
            auto vec3 = a.get<mv::QuantizationParams>().getMin();
            if (vec3.size() > 0)
            {
                for (std::size_t i = 0; i < vec3.size() - 1; ++i)
                    output += std::to_string(vec3[i]) + ", ";
                output += std::to_string(vec3.back());
            }
            output += "},{";
            auto vec4 = a.get<mv::QuantizationParams>().getMax();
            if (vec4.size() > 0)
            {
                for (std::size_t i = 0; i < vec4.size() - 1; ++i)
                    output += std::to_string(vec4[i]) + ", ";
                output += std::to_string(vec4.back());
            }
            return output + "}}";
        }

        MV_REGISTER_ATTR(mv::QuantizationParams)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setToStringFunc(toString)
            .setToLongStringFunc(toLongString)
            .setTypeTrait("large");
    }

}
