#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/kmb/barrier_deps.hpp"

namespace mv
{

    namespace attr_barrier_deps
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        auto bdep = a.get<BarrierDependencies>();
        json::Object result;

        json::Array updateArray;
        for (auto u: bdep.getUpdate())
            updateArray.append(static_cast<long long>(u));

        json::Array waitArray;
        for (auto w: bdep.getWait())
            waitArray.append(static_cast<long long>(w));

        result.emplace("wait", waitArray);
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


    }


    namespace attr {
	    MV_REGISTER_ATTR(BarrierDependencies)
		.setToJSONFunc(attr_barrier_deps::toJSON)
		.setFromJSONFunc(attr_barrier_deps::fromJSON)
		.setToStringFunc(attr_barrier_deps::toString);
    }

}
