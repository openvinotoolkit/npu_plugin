#ifndef MV_JSON_OBJECT_HPP_
#define MV_JSON_OBJECT_HPP_

#include <map>
#include <string>
#include <memory>
#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class Object : public Value
        {

            friend class Value;
            std::map<std::string, std::unique_ptr<Value>> members_;

            void deepCopyMembers_(const std::map<std::string, std::unique_ptr<Value>>& input);

        public:

            Object();
            Object(const Object& other);
            Object(Object& owner, const std::string& key);
            bool emplace(const std::string& key, float value);
            bool emplace(const std::string& key, int value);
            bool emplace(const std::string& key, const std::string& value);
            bool emplace(const std::string& key, bool value);
            bool emplace(const std::string& key, const Object& value);
            bool emplace(const std::string& key);
            void erase(const std::string& key);
            unsigned size() const;
            Value& operator[](const std::string& key) override;
            std::string stringify() const override;
            Object& operator=(const Object& other);

        };  

    }

}

#endif // MV_JSON_OBJECT_HPP_