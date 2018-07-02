#ifndef MV_JSON_VALUE_HPP_
#define MV_JSON_VALUE_HPP_

#include <string>
#include <map>
#include "include/mcm/base/json/exception/value_error.hpp"

namespace mv
{

    namespace json
    {

        class NumberInteger;
        class NumberFloat;
        class String;
        class Object;
        class JSON;

        enum class JSONType
        {
            Unknown,
            Object,
            Array,
            String,
            NumberInteger,
            NumberFloat,
            Bool,
            Null
        };
    
        class Value
        {

            static const std::map<JSONType, std::string> typeString_;
            JSONType valueType_;
            Object* owner_;
            std::string key_;

        public:

            Value(Object& owner, const std::string& key, JSONType valueType);
            Value();
            virtual ~Value() = 0;
            virtual explicit operator float&();
            virtual explicit operator int&();
            virtual explicit operator std::string&();
            virtual explicit operator bool&();
            virtual Value& operator=(float value);
            virtual Value& operator=(int value);
            virtual Value& operator=(const std::string& value);
            virtual Value& operator=(bool value);
            virtual Value& operator[](const std::string& key);
            virtual std::string stringify() const = 0;

            template <class T_value>
            T_value& get()
            {
                
                return (T_value&)(*this);

            }

            
        };  

    }

}

#endif // MV_JSON_VALUE_HPP_