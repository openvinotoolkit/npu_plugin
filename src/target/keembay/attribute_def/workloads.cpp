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
       
    }

    static Attribute fromJSON(const json::Value& v)
    {
        
    }

    static std::string toString(const Attribute& a)
    {
      //Complete this
      std::string output = "{";
      auto s = a.get<Workloads>();
      for (std::size_t i = 0; i < s.nWorkloads() - 1; ++i) {
          output += std::to_string(s[i].MaxX) + ", ";
          output += std::to_string(s[i].MinX) + ", ";
          output += std::to_string(s[i].MaxY) + ", ";
          output += std::to_string(s[i].MinY) + ", ";
          output += std::to_string(s[i].MaxZ) + ", ";
          output += std::to_string(s[i].MinZ) + ", ";
      }
      output += "}";
    }

    MV_REGISTER_ATTR(Workloads)
        .setToJSONFunc(toJSON)
        .setFromJSONFunc(fromJSON)
        .setToStringFunc(toString);

    }

}
