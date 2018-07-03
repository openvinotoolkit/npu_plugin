#ifndef MV_JSON_OBJECT_HPP_
#define MV_JSON_OBJECT_HPP_

#include <unordered_map>
#include <string>
#include <memory>
#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class Object : public ValueContent
        {

            std::unordered_map<std::string, Value> members_;

        public:

            Object();
            Object(const Object& other);
            Object(std::initializer_list<std::pair<const std::string, Value>> l);
            bool emplace(const std::string& key, const Value& value);
            void erase(const std::string& key);
            unsigned size() const;
            Value& operator[](const std::string& key);
            Object& operator=(const Object& other);
            std::string stringify() const override;

        };  

    }

}

#endif // MV_JSON_OBJECT_HPP_