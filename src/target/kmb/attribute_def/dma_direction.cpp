#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/kmb/dma_direction.hpp"

namespace mv
{

    namespace attr_dma_direction
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        auto d = a.get<DmaDirection>();
        return json::Value(d.toString());
    }

    static Attribute fromJSON(const json::Value& v)
    {
        if (v.valueType() != json::JSONType::String)
            throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                " to mv::DmaDirection");

        return DmaDirection(v.get<std::string>());
    }

    static std::string toString(const Attribute& a)
    {
        return "DmaDirection::" + a.get<DmaDirection>().toString();
    }


    }

    namespace attr {
	    MV_REGISTER_ATTR(DmaDirection)
		.setToJSONFunc(attr_dma_direction::toJSON)
		.setFromJSONFunc(attr_dma_direction::fromJSON)
		.setToStringFunc(attr_dma_direction::toString);

    }

}
