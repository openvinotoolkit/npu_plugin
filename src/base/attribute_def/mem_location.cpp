#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/tensor/tensor.hpp"


namespace mv
{
    namespace attr_mem_location
    {
        static mv::json::Value toJSON(const Attribute& a)
        {
            auto& elem = a.get<Tensor::MemoryLocation>();
            return json::Value(elem.toString());
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if( v.valueType() == json::JSONType::NumberInteger )
                return( v.get<long long>());
            else if (v.valueType() == json::JSONType::String )
                return( v.get<std::string>());
            else
                throw AttributeError(v," Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) + "to Location");
        }

        static std::string toString(const Attribute& a)
        {
            auto& elem = a.get<Tensor::MemoryLocation>();
            return elem.toString();
        }


    }

    namespace attr {
        MV_REGISTER_ATTR(Tensor::MemoryLocation)
            .setToJSONFunc(attr_mem_location::toJSON)
            .setFromJSONFunc(attr_mem_location::fromJSON)
            .setToStringFunc(attr_mem_location::toString);
    }
}
