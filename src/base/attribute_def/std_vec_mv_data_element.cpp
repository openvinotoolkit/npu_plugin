#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/data_element.hpp"
#include <vector>
#include <string>

namespace mv
{

    namespace attr_std_vec_mv_data
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            std::cout << "XXXXXXXXXXXXXXXXXXXXXXXX JUST BEFORE " << std::endl;
            auto vec = a.get<std::vector<mv::DataElement>>();
            std::cout << "XXXXXXXXXXXXXXXXXXXXXXXX JUST AFTER " << std::endl;
            for (std::size_t i = 0; i < vec.size(); ++i)
            {
                output.append(vec[i].isDouble());
                if (vec[i].isDouble())
                    output.append((double) vec[i]);
                else
                    output.append(static_cast<long long>((int64_t) vec[i]));
            }
            return output;
        }


        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to std::vector<mv::DataElement>");

            if (v.size() % 2 != 0)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to std::vector<mv::DataElement>, size is not a multiple of 2");
            std::vector<mv::DataElement> output;
            bool isDouble;

            for (std::size_t i = 0; i < v.size(); i+=2)
            {
                if (v[i].valueType() != json::JSONType::Bool)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i].valueType()) +
                    " to bool (during the conversion to vector<mv::DataElement>)");
                isDouble = v[i].get<bool>();

                if (v[i+1].valueType() != json::JSONType::NumberInteger)
                    throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v[i+1].valueType()) +
                    " to NumberInterger (during the conversion to vector<mv::DataElement>)");
                output.push_back(DataElement(isDouble, static_cast<int64_t>(v[i+1].get<long long>())));
            }

            return output;
        }

        static std::string toLongString(const Attribute& a)
        {
            std::string output = "{[";
            auto vec = a.get<std::vector<mv::DataElement>>();
            if (vec.size() > 0)
            {
                for (std::size_t i = 0; i < vec.size() - 1; ++i)
                    output += "\"" + static_cast<std::string>(vec[i]) + "\", ";
            }
            output += "]}";
            return output;
        }

        static std::string toString(const Attribute& a)
        {
            std::string output = "(" + std::to_string(a.get<std::vector<mv::DataElement>>().size()) + ")";
            return output;
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(std::vector<mv::DataElement>)
            .setToJSONFunc(attr_std_vec_mv_data::toJSON)
            .setFromJSONFunc(attr_std_vec_mv_data::fromJSON)
            .setToStringFunc(attr_std_vec_mv_data::toString);
    }

}
