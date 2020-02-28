#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>

namespace mv
{

    namespace attr_std_vector_uint8_t
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<uint8_t>>();
            for (std::size_t i = 0; i < vec.size(); ++i)
                output.append(static_cast<long long>(vec[i]));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                                        " to std::vector<uint8_t>");

            std::vector<uint8_t> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::NumberInteger)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                                            " to uint8_t (during the conversion to std::vector<uint8_t>)");
                output.push_back(static_cast<uint8_t>(v[i].get<long long>()));
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "(" + std::to_string(a.get<std::vector<uint8_t>>().size()) + ")";
            return output;
        }

        static std::string toLongString(const Attribute& a)
        {
            auto vec = a.get<std::vector<uint8_t>>();
            auto output = std::accumulate(vec.begin(), vec.end(), std::string("{"),
                                          [](std::string const& left, uint8_t value) {
                                              return left + ", " + std::to_string(value);
                                          });
            output += "}";
            return output;
        }

    }

    namespace attr {

        MV_REGISTER_ATTR(std::vector<uint8_t>)
                .setToJSONFunc(attr_std_vector_uint8_t::toJSON)
                .setFromJSONFunc(attr_std_vector_uint8_t::fromJSON)
                .setToStringFunc(attr_std_vector_uint8_t::toString);

    }

}