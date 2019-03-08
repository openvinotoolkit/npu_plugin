#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/keembay/barrier_definition.hpp"

namespace mv
{

    namespace attr
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
       auto b = a.get<Barrier>();
       return json::Value(b.toString());
    }

    static Attribute fromJSON(const json::Value&)
    {
        
    }

    static std::string toString(const Attribute& a)
    {
       return a.get<Barrier>().toString();
    }

    MV_REGISTER_ATTR(Barrier)
        .setToJSONFunc(toJSON)
        .setFromJSONFunc(fromJSON)
        .setToStringFunc(toString);

    }

}
