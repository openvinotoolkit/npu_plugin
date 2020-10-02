#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>
#include <functional>
#include "schema/graphfile/graphfile_generated.h"

namespace mv {
namespace attr_lambda_function {
static mv::json::Value toJSON(const Attribute&) {
    return json::Value("function(graphFileInstance)");
}

static Attribute fromJSON(const json::Value&) {
    return 0;
}

static std::string toString(const Attribute& a) {
    return "std::function<void(MVCNN::GraphFileT&)>()";
}

static std::vector<uint8_t> toBinary(const Attribute&) {
    return {0x0};
}
}  // namespace attr_lambda_function

namespace attr {
    MV_REGISTER_ATTR(std::function<void(MVCNN::GraphFileT&)>)
        .setToJSONFunc(attr_lambda_function::toJSON)
        .setFromJSONFunc(attr_lambda_function::fromJSON)
        .setToStringFunc(attr_lambda_function::toString)
        .setToBinaryFunc(attr_lambda_function::toBinary);
}
}  // namespace mv
