#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/exception/attribute_error.hpp"
#include "include/mcm/base/attribute.hpp"
#include <vector>
#include <functional>

namespace mv {
namespace attr_lambda_function {
static mv::json::Value toJSON(const Attribute&) {
    return json::Value("function(graphFileInstance)");
}

static Attribute fromJSON(const json::Value&) {
    return 0;
}

static std::string toString(const Attribute& a) {
    return "std::function<void(void*)>()";
}

static std::vector<uint8_t> toBinary(const Attribute&) {
    return {0x0};
}
}  // namespace attr_lambda_function

namespace attr {
    // FIXME use MVCNN::GraphFileT reference here instead of void pointer
    // for some reason, it's impossible to include graphfile_generated.h
    MV_REGISTER_ATTR(std::function<void(void* graphFileInstance)>)
        .setToJSONFunc(attr_lambda_function::toJSON)
        .setFromJSONFunc(attr_lambda_function::fromJSON)
        .setToStringFunc(attr_lambda_function::toString)
        .setToBinaryFunc(attr_lambda_function::toBinary);
}
}  // namespace mv
