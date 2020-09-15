#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>
#include <string>

namespace mv
{

    namespace attr_std_vector_std_string
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<std::string>>();
            for (std::size_t i = 0; i < vec.size(); ++i)
                output.append(vec[i]);
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to std::vector<std::string>");
            
            std::vector<std::string> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::String)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) + 
                    " to std::size_t (during the conversion to vector<std::string>)");
                output.push_back(v[i].get<std::string>());
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "{";
            auto vec = a.get<std::vector<std::string>>();
            if (vec.size() > 0)
            {
                for (std::size_t i = 0; i < vec.size() - 1; ++i)
                    output += "\"" + vec[i] + "\", ";
                output += "\"" + *vec.rbegin() + "\"";
            }
            output += "}";
            return output;
        }


    }

    namespace attr {

        MV_REGISTER_ATTR(std::vector<std::string>)
            .setToJSONFunc(attr_std_vector_std_string::toJSON)
            .setFromJSONFunc(attr_std_vector_std_string::fromJSON)
            .setToStringFunc(attr_std_vector_std_string::toString);
    }

}
