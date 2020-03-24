#include <sstream>
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

namespace mv
{

    namespace attr_quantization_params
    {

        template <typename T>
        static std::string toStringWithPrecision(const T val, const int precision = 15)
        {
            std::ostringstream outstream;
            outstream.precision(precision);
            outstream << std::fixed << val;
            return outstream.str();
        }

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
            auto quant_param = a.get<mv::QuantizationParams>();
            std::string output;
            if (quant_param.hasAttr("mult"))
            {
                output = "(" + std::to_string(quant_param.getZeroPoint().size()) + ", " +
                std::to_string(quant_param.getScale().size()) + ", " +
                std::to_string(quant_param.getMin().size()) + ", " +
                std::to_string(quant_param.getMax().size()) + ", " +
                std::to_string(quant_param.getShift().size()) + ", " +
                std::to_string(quant_param.getMult().size()) + ")";
            }
            else
            {
                output = "(" + std::to_string(a.get<mv::QuantizationParams>().getZeroPoint().size()) + ", " +
                std::to_string(quant_param.getScale().size()) + ", " +
                std::to_string(quant_param.getMin().size()) + ", " +
                std::to_string(quant_param.getMax().size()) + ")";
            }
            return output;
        }

        static std::string toLongString(const Attribute& a)
        {
            std::string output = "{{";
            auto quant_param = a.get<mv::QuantizationParams>();
            auto vec1 = quant_param.getZeroPoint();
            if (vec1.size() > 0)
            {
                for (std::size_t i = 0; i < vec1.size() - 1; ++i)
                    output += toStringWithPrecision(vec1[i]) + ", ";
                output += toStringWithPrecision(vec1.back());
            }
            output += "},{";
            auto vec2 = quant_param.getScale();
            if (vec2.size() > 0)
            {
                for (std::size_t i = 0; i < vec2.size() - 1; ++i)
                    output += toStringWithPrecision(vec2[i]) + ", ";
                output += toStringWithPrecision(vec2.back());
            }
            output += "},{";
            auto vec3 = quant_param.getMin();
            if (vec3.size() > 0)
            {
                for (std::size_t i = 0; i < vec3.size() - 1; ++i)
                    output += toStringWithPrecision(vec3[i]) + ", ";
                output += toStringWithPrecision(vec3.back());
            }
            output += "},{";
            auto vec4 = quant_param.getMax();
            if (vec4.size() > 0)
            {
                for (std::size_t i = 0; i < vec4.size() - 1; ++i)
                    output += toStringWithPrecision(vec4[i]) + ", ";
                output += toStringWithPrecision(vec4.back());
            }
            if (quant_param.hasAttr("mult"))
            {
                output += "},{";
                auto vec5 = quant_param.getShift();
                    if (vec5.size() > 0)
                    {
                        for (std::size_t i = 0; i < vec5.size() - 1; ++i)
                            output += toStringWithPrecision(vec5[i]) + ", ";
                        output += toStringWithPrecision(vec5.back());
                    }
                output += "},{";
                auto vec6 = quant_param.getMult();
                if (vec6.size() > 0)
                {
                    for (std::size_t i = 0; i < vec6.size() - 1; ++i)
                        output += toStringWithPrecision(vec6[i]) + ", ";
                    output += toStringWithPrecision(vec6.back());
                }
            }
            return output + "}}";
        }

    }

    namespace attr {
        MV_REGISTER_ATTR(mv::QuantizationParams)
            .setToJSONFunc(attr_quantization_params::toJSON)
            .setFromJSONFunc(attr_quantization_params::fromJSON)
            .setToStringFunc(attr_quantization_params::toString)
            .setToLongStringFunc(attr_quantization_params::toLongString)
            .setTypeTrait("large");
    }

}
