#ifndef MV_JSON_OBJECT_HPP_
#define MV_JSON_OBJECT_HPP_

#include <map>
#include <string>
#include <memory>
#include <type_traits>
#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/json/number_integer.hpp"
#include "include/mcm/base/json/number_float.hpp"
#include "include/mcm/base/json/string.hpp"
#include "include/mcm/base/json/null.hpp"

namespace mv
{

    namespace json
    {

        

        class Object
        {

            std::map<std::string, std::unique_ptr<Value>> members_;

        public:

            Object();
            bool emplace(const std::string& key, float value);
            bool emplace(const std::string& key, unsigned value);
            bool emplace(const std::string& key, int value);
            bool emplace(const std::string& key, const std::string& value);
            bool emplace(const std::string& key, bool value);
            bool emplace(const std::string& key);
            //bool erase(const std::string& name);
            //std::reference_wrapper<Value> get(const std::string& name);
            //std::map<std::string, Value>::iterator begin();
            //std::map<std::string, Value>::iterator end();
            unsigned size() const;

            Value& operator[](const std::string& key)
            {

                if (members_.find(key) == members_.end())
                {
                    throw std::logic_error("Key not found");
                }

                return *members_[key].get();

            }

        };  

    }

}

#endif // MV_JSON_OBJECT_HPP_