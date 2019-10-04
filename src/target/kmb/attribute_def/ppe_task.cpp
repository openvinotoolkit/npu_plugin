#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"

namespace mv
{

    namespace attr_ppe_task
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        auto d = a.get<PPETask>();
        return json::Value(d.toString());
    }

    static Attribute fromJSON(const json::Value& v)
    {
        if (v.valueType() != json::JSONType::String)
            throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                " to mv::DType");

        return PPETask(v);
    }

    static std::string toString(const Attribute& a)
    {
        return "PPETask::" + a.get<PPETask>().toString();
    }


    }

    namespace attr {
	    MV_REGISTER_ATTR(PPETask)
		.setToJSONFunc(attr_ppe_task::toJSON)
		.setFromJSONFunc(attr_ppe_task::fromJSON)
		.setToStringFunc(attr_ppe_task::toString);

    }

}
