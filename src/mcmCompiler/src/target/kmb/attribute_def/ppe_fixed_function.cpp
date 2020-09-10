#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/target/kmb/ppe_fixed_function.hpp"

namespace mv
{

    namespace attr_ppe_fixed_function
    {

    static mv::json::Value toJSON(const Attribute& a)
    {
        auto d = a.get<PPEFixedFunction>();
        json::Object toBuild;
        toBuild["lowClamp"] = json::Value(static_cast<long long>(d.getLowClamp()));
        toBuild["highClamp"] = json::Value(static_cast<long long>(d.getHighClamp()));
        auto layers = d.getLayers();
        toBuild["layers"] = json::Array();
        for(auto layer : layers)
            toBuild["layers"].append(json::Value(layer.toString()));

        return json::Value(toBuild);
    }

    static Attribute fromJSON(const json::Value& v)
    {
        if (v.valueType() != json::JSONType::String)
            throw AttributeError(v, "Unable to convert JSON value of type " + json::Value::typeName(v.valueType()) +
                " to mv::DType");

        int lowClamp = v["lowClamp"].get<long long>();
        int highClamp = v["highClamp"].get<long long>();

        PPEFixedFunction toReturn(lowClamp, highClamp);
        auto layers = v["layers"];
        auto n = layers.size();
        for(unsigned i = 0; i < n; ++i)
            toReturn.addLayer(PPELayerType(layers[i].get<std::string>()));
        return toReturn;
    }

    static std::string toString(const Attribute& a)
    {
        return "PPEFixedFunction::" + a.get<PPEFixedFunction>().toString();
    }


    }


    namespace attr {
	    MV_REGISTER_ATTR(PPEFixedFunction)
		.setToJSONFunc(attr_ppe_fixed_function::toJSON)
		.setFromJSONFunc(attr_ppe_fixed_function::fromJSON)
		.setToStringFunc(attr_ppe_fixed_function::toString);
    }

}
