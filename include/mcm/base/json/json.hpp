#ifndef MV_JSON_JSON_HPP_
#define MV_JSON_JSON_HPP_

#include <map>
#include <string>
#include <memory>
#include <type_traits>
#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/bool.hpp"
#include "include/mcm/base/json/null.hpp"

namespace mv
{

    namespace json
    {

        class JSON
        {
            
            friend class Value;
            std::map<std::string, std::unique_ptr<Value>> members_;

        public:

            JSON();
            bool emplace(const std::string& key, float value);
            bool emplace(const std::string& key, int value);
            bool emplace(const std::string& key, const std::string& value);
            bool emplace(const std::string& key, bool value);
            bool emplace(const std::string& key);
            void erase(const std::string& key);
            unsigned size() const;
            //Value& operator[](const std::string& key);
            std::string stringify() const;

        };  

    }

}

#endif // MV_JSON_JSON_HPP_