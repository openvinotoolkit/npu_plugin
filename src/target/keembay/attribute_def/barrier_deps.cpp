#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/keembay/barrier_deps.hpp"

namespace mv
{

    namespace attr
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        auto bdep = a.get<BarrierDependencies>();
        json::Object result;

        json::Array updateArray;
        for (auto u: bdep.getUpdate())
            updateArray.append(static_cast<long long>(u));

        result.emplace("wait", static_cast<long long>(bdep.getWait()));
        result.emplace("update", updateArray);

        return result;
    }

    static Attribute fromJSON(const json::Value&)
    {
        
    }

    static std::string toString(const Attribute& a)
    {
       return a.get<BarrierDependencies>().toString();
    }

    MV_REGISTER_ATTR(BarrierDependencies)
        .setToJSONFunc(toJSON)
        .setFromJSONFunc(fromJSON)
        .setToStringFunc(toString);

    }

}
