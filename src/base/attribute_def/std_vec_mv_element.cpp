#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/element.hpp"
#include <vector>
#include <string>

namespace mv
{

    namespace attr_std_vec_mv
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<mv::Element>>();
            for (std::size_t i = 0; i < vec.size(); ++i)
                output.append(vec[i].toJSON());
            return output;
        }

        static mv::json::Value toSimplifiedJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<mv::Element>>();
            for (std::size_t i = 0; i < vec.size(); ++i)
                output.append(vec[i].toJSON(true));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to std::vector<mv::Element>");

            std::vector<mv::Element> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::Object)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to mv::Element (during the conversion to vector<mv::Element>)");

                output.push_back(Element(v[i]));
            }

            return output;
        }

        static Attribute fromSimplifiedJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to std::vector<mv::Element>");

            std::vector<mv::Element> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::Object)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to mv::Element (during the conversion to vector<mv::Element>)");

                output.push_back(Element(v[i], true));
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "{[";
            auto vec = a.get<std::vector<mv::Element>>();
            if (vec.size() > 0)
            {
                for (std::size_t i = 0; i < vec.size() - 1; ++i)
                    output += "\"" + vec[i].toString() + "\", ";
            }
            output += "]}";
            return output;
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(std::vector<mv::Element>)
            .setToJSONFunc(attr_std_vec_mv::toJSON)
            .setFromJSONFunc(attr_std_vec_mv::fromJSON)
            .setToSimplifiedJSONFunc(attr_std_vec_mv::toSimplifiedJSON)
            .setFromSimplifiedJSONFunc(attr_std_vec_mv::fromSimplifiedJSON)
            .setToStringFunc(attr_std_vec_mv::toString);


    }

}
