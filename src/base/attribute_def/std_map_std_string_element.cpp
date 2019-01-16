#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/element.hpp"
#include <string>
#include <map>

namespace mv
{

    namespace attr
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

            // Declare the object to be returned
            std::map<std::string, Element> output;

            // // Get the keys in the input json object
            // std::vector<std::string> keys = v.getKeys();

            // // populate the output with values passed in
            // for (auto key: keys) {
            //     output[key] = v[key].get<Element>();
            // }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "{";
            auto map = a.get<std::map<std::string, Element>>();
            if (map.size() > 0)
            {
                for (auto item: map) {
                    output += "\"" + item.first + "\"";
                    output += item.second.toString();
                }
            }
            output += "}";
            return output;
        }

        MV_REGISTER_ATTR(std::map<std::string COMMA Element>)
            .setToJSONFunc(toJSON)      // not done yet
            .setFromJSONFunc(fromJSON)  // not done yet
            .setToStringFunc(toString); //done

    }

}