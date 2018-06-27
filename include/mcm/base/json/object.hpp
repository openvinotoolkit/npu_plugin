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

namespace mv
{

    namespace json
    {

        

        class Object
        {

            std::string name_;
            std::map<std::string, std::unique_ptr<Value>> members_;
            bool addMember_(const std::string& name, std::unique_ptr<Value>& value);

        public:

            Object();
            //bool emplace(const std::string& name, const Number& value);
            //bool erase(const std::string& name);
            //std::reference_wrapper<Value> get(const std::string& name);
            //std::map<std::string, Value>::iterator begin();
            //std::map<std::string, Value>::iterator end();
            unsigned size() const;
            
            template <class T_member>
            bool emplace(const std::string& key, const T_member& value)
            {
                
                if (members_.find(key) != members_.end())
                    return false;
                
                if (std::is_integral<T_member>::value)
                {
                    std::unique_ptr<NumberInteger> valuePtr(new NumberInteger(value));
                    members_.emplace(key, std::unique_ptr<Value>(std::move(valuePtr)));
                    return true;
                }

                if (std::is_floating_point<T_member>::value)
                {
                    std::unique_ptr<NumberFloat> valuePtr(new NumberFloat(value));
                    members_.emplace(key, std::unique_ptr<Value>(std::move(valuePtr)));
                    return true;
                }

                if (std::is_convertible<T_member, std::string>::value)
                {
                    std::unique_ptr<String> valuePtr(new String(value));
                    members_.emplace(key, std::unique_ptr<Value>(std::move(valuePtr)));
                    return true;
                }

                return false;

            }

            Value& operator[](const std::string& key)
            {

                if (members_.find(key) == members_.end())
                    throw std::out_of_range("Unable to find member " + key);

                return *members_[key].get();
            }

        };  

    }

}

#endif // MV_JSON_OBJECT_HPP_