#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/tensor_info.hpp"
#include "include/mcm/utils/json_serialization.hpp"

namespace mv
{

    namespace attr_tensor
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            json::Array output;
            auto tensor = a.get<TensorInfo>();
            output.append(argToJSON(tensor.shape()));
            output.append(argToJSON(tensor.type()));
            output.append(argToJSON(tensor.order()));
            return output;
        }

        static Attribute fromJSON(const json::Value& v)
        {
           if (v.valueType() != json::JSONType::Array || v.size() != 3)
              throw AttributeError(v, "Unable to convert JSON value of type " +
                  json::Value::typeName(v.valueType()) + " to mv::TensorInfo");

           auto shape = argFromJSON<Shape>(v[0]);
           auto type = argFromJSON<DType>(v[1]);
           auto order = argFromJSON<Order>(v[2]);

           return mv::TensorInfo{std::move(shape), std::move(type), std::move(order)};
        }

        static std::string toString(const Attribute& a)
        {
            return a.get<TensorInfo>().toString();
        }

    }

    namespace attr {

        MV_REGISTER_ATTR(TensorInfo)
            .setToJSONFunc(attr_tensor::toJSON)
            .setFromJSONFunc(attr_tensor::fromJSON)
            .setToStringFunc(attr_tensor::toString);

    }

}
