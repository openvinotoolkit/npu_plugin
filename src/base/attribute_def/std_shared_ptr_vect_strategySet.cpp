#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"

namespace mv
{
    //todo:: do not replicate the StrategySet definition
    using StrategySet = std::unordered_map<std::string,Attribute>;
    namespace attr
    {

        static mv::json::Value toJSON(const Attribute& a)
        {
            auto elem = a.get<std::shared_ptr<std::vector<StrategySet>>>();
            //we do not want to print this, eventually, only the number of strategies.
            return json::Value(static_cast<long long>(elem->size()));
        }

        static Attribute fromJSON(const json::Value& v)
        {
            if (v.valueType() != json::JSONType::NumberInteger)
                throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                    " to std::size_t");

            //again, we do not want to specify this from JSON. We only want
            auto newStrategy = std::make_shared<std::vector<StrategySet>>(v.get<long long>());

            return newStrategy;
        }

        static std::string toString(const Attribute& a)
        {
            auto elem = a.get<std::shared_ptr<std::vector<StrategySet>>>();
            return std::to_string(elem->size());
        }

        MV_REGISTER_DUPLICATE_ATTR(std::shared_ptr<std::vector<StrategySet>>)
            .setToJSONFunc(toJSON)
            .setFromJSONFunc(fromJSON)
            .setToStringFunc(toString);
    }

}
