#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>
#include <string>

namespace mv
{

    namespace attr_std_set_std_string
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::set<std::string>>();
            for(std::string key : vec)
                output.append(key);
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to std::vector<std::string>");
            
            std::set<std::string> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::String)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) + 
                    " to std::size_t (during the conversion to vector<std::string>)");
                output.insert(v[i].get<std::string>());
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "{";
            auto vec = a.get<std::set<std::string>>();
            for(std::string key : vec)
                output += "\"" + key + "\", ";
            output += "}";
            return output;
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(std::set<std::string>)
            .setToJSONFunc(attr_std_set_std_string::toJSON)
            .setFromJSONFunc(attr_std_set_std_string::fromJSON)
            .setToStringFunc(attr_std_set_std_string::toString);
    }

}
