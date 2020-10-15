#include "include/mcm/base/json/json.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include <fstream>

void mv::save(const json::Object& obj, const std::string& path)
{

    utils::validatePath(path);
    std::ofstream out(path);
    out << obj.stringifyPretty();
    out.close();

}

void mv::save(const json::Value& obj, const std::string& path)
{

    if (obj.valueType() != json::JSONType::Object)
        throw ArgumentError("JSON:save", "obj", "type", "Has to be mv::json::Object");
    
    save(obj.get<json::Object>(), path);
    
}
