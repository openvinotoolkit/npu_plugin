#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>

namespace mv
{

    namespace attr_std_vector_double
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<double>>();
            for (std::size_t i = 0; i < vec.size(); ++i)
                output.append(static_cast<double>(vec[i]));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to std::vector<double>");
            
            std::vector<double> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::NumberFloat)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) + 
                    " to double (during the conversion to std::vector<double>)");
                output.push_back(v[i].get<double>());
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "(" + std::to_string(a.get<std::vector<double>>().size()) + ")";
            return output;
        }

        static std::string toLongString(const Attribute& a)
        {
            std::string output = "{";
            auto vec = a.get<std::vector<double>>();
            if (vec.size() > 0)
            {
                for (std::size_t i = 0; i < vec.size() - 1; ++i)
                    output += std::to_string(vec[i]) + ", ";
                output += std::to_string(vec.back());
            }
            return output + "}";
        }

    }

    namespace attr {
        MV_REGISTER_ATTR(std::vector<double>)
            .setToJSONFunc(attr_std_vector_double::toJSON)
            .setFromJSONFunc(attr_std_vector_double::fromJSON)
            .setToStringFunc(attr_std_vector_double::toString)
            .setToLongStringFunc(attr_std_vector_double::toLongString)
            .setTypeTrait("large");


    }

}
