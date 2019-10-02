#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/kmb/workloads.hpp"

namespace mv
{

    namespace attr_workloads
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        auto w = a.get<Workloads>();
        return json::Value(w.toString());
    }

    static Attribute fromJSON(const json::Value& v)
    {
        
    }

    static std::string toString(const Attribute& a)
    {
       return a.get<Workloads>().toString();
    }

    static std::string toLongString(const Attribute& a)
    {
        return a.get<Workloads>().toLongString();
    }

    }

    namespace attr {
	    MV_REGISTER_ATTR(Workloads)
		.setToJSONFunc(attr_workloads::toJSON)
		.setFromJSONFunc(attr_workloads::fromJSON)
		.setToStringFunc(attr_workloads::toString)
		.setToLongStringFunc(attr_workloads::toLongString)
		.setTypeTrait("large");
    }

}
