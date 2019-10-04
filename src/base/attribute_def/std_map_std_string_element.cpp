#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/element.hpp"
#include <string>
#include <map>

namespace mv
{

    namespace attr_std_map_std_string
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Object output;
            auto map = a.get<std::map<std::string, Element>>();

            for (auto map_elem: map) {
                output.emplace(map_elem.first, map_elem.second.toJSON());
            }

            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Object)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to std::map<std::string, Element>");

            std::map<std::string, Element> output;
            std::vector<std::string> keys = v.getKeys();
            for (auto key: keys) {
                output.emplace(key, Element(v[key]));
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "{";
            output += "\n";
            auto map = a.get<std::map<std::string, Element>>();
            if (map.size() > 0)
            {
                for (auto item: map) {
                    output += "\"" + item.first + "\"";
                    output += "{";
                    output += item.second.toString();
                    output += "}";
                    output += "\n";
                }
            }
            output += "}";
            return output;
        }

    }

    namespace attr {
        MV_REGISTER_ATTR(std::map<std::string COMMA Element>)
            .setToJSONFunc(attr_std_map_std_string::toJSON)
            .setFromJSONFunc(attr_std_map_std_string::fromJSON)
            .setToStringFunc(attr_std_map_std_string::toString);

    }

}
