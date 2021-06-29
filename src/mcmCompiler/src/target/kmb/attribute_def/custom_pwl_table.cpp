#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/kmb/custom_pwl_table.hpp"

namespace mv
{

    namespace attr_custom_pwl_table
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        auto d = a.get<PWLTableType>();
        json::Object toBuild;
        toBuild["activation"] = json::Value(d.activation);
        toBuild["dtype"] = json::Value(d.dtype.toString());

        return json::Value(toBuild);
    }

    static Attribute fromJSON(const json::Value& v)
    {
        PWLTableType toReturn;

        toReturn.activation = v["activation"].get<std::string>();
        toReturn.dtype = mv::DType(v["dtype"].get<std::string>());

        return toReturn;
    }

    static std::string toString(const Attribute& a)
    {
        PWLTableType pt = a.get<PWLTableType>();

        return "PWLTableType(" + pt.activation + ", " + pt.dtype.toString() + ")";
    }


    }


    namespace attr {
        MV_REGISTER_SIMPLE_ATTR(PWLTableType)
               .setToJSONFunc(attr_custom_pwl_table::toJSON)
               .setFromJSONFunc(attr_custom_pwl_table::fromJSON)
               .setToStringFunc(attr_custom_pwl_table::toString);
    }

}
