#ifndef MV_JSON_JSON_HPP_
#define MV_JSON_JSON_HPP_

#include "include/mcm/base/json/object.hpp"
#include "include/mcm/base/json/array.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/null.hpp"

namespace mv
{

    void save(const json::Object& obj, const std::string& path);
    void save(const json::Value& obj, const std::string& path);
    
}

#endif // MV_JSON_JSON_HPP_