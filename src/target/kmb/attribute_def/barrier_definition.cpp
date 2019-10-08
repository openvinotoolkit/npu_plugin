#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/kmb/barrier_definition.hpp"

namespace mv
{

    namespace attr_barrier_definition
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        json::Object result;
        auto b = a.get<Barrier>();

        json::Array producerArray;
        for (auto p: b.getProducers())
            producerArray.append(p);

        json::Array consumerArray;
        for (auto p: b.getConsumers())
            consumerArray.append(p);

        result.emplace("group", static_cast<long long> (b.getGroup()));
        result.emplace("index", static_cast<long long> (b.getIndex()));
        result.emplace("numProducers", static_cast<long long> (b.getNumProducers()));
        result.emplace("numConsumers", static_cast<long long> (b.getNumConsumers()));
        result.emplace("producers", producerArray);
        result.emplace("consumers", consumerArray);

        return result;
    }

    static Attribute fromJSON(const json::Value&)
    {
        return Barrier();
    }

    static std::string toString(const Attribute& a)
    {
       return a.get<Barrier>().toString();
    }


    }

    namespace attr {
	    MV_REGISTER_ATTR(Barrier)
		.setToJSONFunc(attr_barrier_definition::toJSON)
		.setFromJSONFunc(attr_barrier_definition::fromJSON)
		.setToStringFunc(attr_barrier_definition::toString);

    }

}
