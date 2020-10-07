#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/tensor_info.hpp"
#include "include/mcm/utils/json_serialization.hpp"
#include <vector>

namespace mv
{

    namespace attr_std_vector_tensor_info
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto vec = a.get<std::vector<mv::TensorInfo>>();
            for (auto &x : vec)
                output.append(argToJSON(x));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::Array)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                                        " to std::vector<mv::TensorInfo>");

            std::vector<mv::TensorInfo> output;
            for (std::size_t i = 0; i < v.size(); ++i)
            {
                output.push_back(argFromJSON<mv::TensorInfo>(v[i]));
            }

            return output;
        }

        static std::string toString(const Attribute& a)
        {
            const auto& tensors = a.get<std::vector<mv::TensorInfo>>();

            auto str = std::string{};
            for (const auto& tensor : tensors) {
                str += tensor.toString() + ", ";
            }
            str.resize(str.size() - 2);

            return str;
        }

    }

    namespace attr {

        MV_REGISTER_ATTR(std::vector<mv::TensorInfo>)
                .setToJSONFunc(attr_std_vector_tensor_info::toJSON)
                .setFromJSONFunc(attr_std_vector_tensor_info::fromJSON)
                .setToStringFunc(attr_std_vector_tensor_info::toString);

    }

}
