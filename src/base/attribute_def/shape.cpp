#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/shape.hpp"

namespace mv
{

    namespace attr_shape
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto s = a.get<Shape>();
            for (std::size_t i = 0; i < s.ndims(); ++i)
                output.append(static_cast<long long>(s[i]));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to mv::Shape");
            
            Shape output(v.size());
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::NumberInteger)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) + 
                    " to std::size_t (during the conversion to mv::Shape)");
                output[i] = static_cast<std::size_t>(v[i].get<long long>());
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "{";
            auto s = a.get<Shape>();
            for (std::size_t i = 0; i < s.ndims() - 1; ++i)
                output += std::to_string(s[i]) + ", ";
            output += std::to_string(s[-1]) + "}";
            return output;
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(Shape)
            .setToJSONFunc(attr_shape::toJSON)
            .setFromJSONFunc(attr_shape::fromJSON)
            .setToStringFunc(attr_shape::toString);
    }

}
