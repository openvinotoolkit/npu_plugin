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
            json::Array output;
            auto q = a.get<QuantizationParams>();

            output.append(static_cast<long long>(q.getZeroPoint()));
            output.append(static_cast<double>(q.getScale()));
            output.append(static_cast<double>(q.getMin()));
            output.append(static_cast<double>(q.getMax()));

            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to mv::QuantizationParams");

            if (v.size() != 4)
                 throw AttributeError(v, "Unable to convert JSON Array to mv::QuantizationParams, size is != to 4");

            std::size_t i = 0;
            if (v[i].valueType() != json::JSONType::NumberInteger)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to int64_t (during the conversion to mv::QuantizationParams)");
            int64_t zero_point = static_cast<int64_t>(v[i].get<long long>());
            i++;

            if (v[i].valueType() != json::JSONType::NumberFloat)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to float (during the conversion to mv::QuantizationParams)");
            float scale = static_cast<float>(v[i].get<double>());
            i++;


            if (v[i].valueType() != json::JSONType::NumberFloat)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to float (during the conversion to mv::QuantizationParams)");
            float min = static_cast<float>(v[i].get<double>());
            i++;


            if (v[i].valueType() != json::JSONType::NumberFloat)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to float (during the conversion to mv::QuantizationParams)");
            float max = static_cast<float>(v[i].get<double>());

            return QuantizationParams(zero_point, scale, min, max);
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