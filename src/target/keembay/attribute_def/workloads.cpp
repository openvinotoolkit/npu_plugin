#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/keembay/workloads.hpp"

namespace mv
{

    namespace attr
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
       return "Workloads:" + a.get<Workloads>().toString();
    }

    static std::string toLongString(const Attribute& a)
    {
        return "Workloads:" + a.get<Workloads>().toLongString();
    }

    MV_REGISTER_ATTR(Workloads)
        .setToJSONFunc(toJSON)
        .setFromJSONFunc(fromJSON)
        .setToStringFunc(toString)
        .setToLongStringFunc(toLongString)
        .setTypeTrait("large");
    }

}
