#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>

namespace mv
{

    namespace attr_std_vector_unsigned
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<unsigned>>();
            for (std::size_t i = 0; i < vec.size(); ++i)
                output.append(static_cast<long long>(vec[i]));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + 
                    " to std::vector<unsigned>");
            
            std::vector<unsigned> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                if (v[i].valueType() != json::JSONType::NumberInteger)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) + 
                    " to unsigned (during the conversion to std::vector<unsigned>)");
                output.push_back(static_cast<unsigned>(v[i].get<long long>()));
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "{";
            auto vec = a.get<std::vector<unsigned>>();
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
                unsigned n;
                uint8_t bytes[sizeof(unsigned)];
            };
            auto vec = a.get<std::vector<unsigned>>();
            std::vector<uint8_t> toReturn(sizeof(unsigned) * vec.size(), 0);
            unsigned i = 0;
            for(auto v: vec)
            {
                Tmp tmp = {v};
                for(unsigned j = 0; j < sizeof(unsigned); ++j)
                    toReturn[i++] = tmp.bytes[j];
            }
            return toReturn;
        }


    }

    namespace attr {

        MV_REGISTER_DUPLICATE_ATTR(std::vector<unsigned>)
            .setToJSONFunc(attr_std_vector_unsigned::toJSON)
            .setFromJSONFunc(attr_std_vector_unsigned::fromJSON)
            .setToStringFunc(attr_std_vector_unsigned::toString)
            .setToBinaryFunc(attr_std_vector_unsigned::toBinary);

    }
}
