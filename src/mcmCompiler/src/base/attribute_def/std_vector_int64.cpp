#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>

namespace mv
{

    namespace attr_std_vector_int64
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<int64_t>>();
            for (std::size_t i = 0; i < vec.size(); ++i)
                output.append(static_cast<long long>(vec[i]));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to std::vector<int64_t>");

            std::vector<int64_t> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::NumberInteger)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to int64_t (during the conversion to std::vector<int64_t>)");
                output.push_back(static_cast<int64_t>(v[i].get<long long>()));
            }

            return output;
        }

       static std::string toString(const Attribute& a)
       {
           std::string output = "(" + std::to_string(a.get<std::vector<int64_t>>().size()) + ")";
           return output;
       }

        static std::string toLongString(const Attribute& a)
        {
            std::string output = "{";
            auto vec = a.get<std::vector<int64_t>>();
            if (vec.size() > 0)
            {
                for (std::size_t i = 0; i < vec.size() - 1; ++i)
                    output += std::to_string(vec[i]) + ", ";
                output += std::to_string(*vec.rbegin());
            }
            output += "}";
            return output;
        }

        static std::vector<uint8_t> toBinary(const Attribute& a)
        {
            union Tmp
            {
                int64_t n;
                uint8_t bytes[sizeof(int64_t)];
            };
            auto vec = a.get<std::vector<int64_t>>();
            std::vector<uint8_t> toReturn(sizeof(int64_t) * vec.size(), 0);
            unsigned i = 0;
            for(auto v: vec)
            {
                Tmp tmp = {v};
                for(unsigned j = 0; j < sizeof(int64_t); ++j)
                    toReturn[i++] = tmp.bytes[j];
            }
            return toReturn;
        }


    }

    namespace attr {

        MV_REGISTER_DUPLICATE_ATTR(std::vector<int64_t>)
            .setToJSONFunc(attr_std_vector_int64::toJSON)
            .setFromJSONFunc(attr_std_vector_int64::fromJSON)
            .setToStringFunc(attr_std_vector_int64::toString)
            .setToLongStringFunc(attr_std_vector_int64::toLongString)
            .setToBinaryFunc(attr_std_vector_int64::toBinary)
            .setTypeTrait("large");


    }

}
