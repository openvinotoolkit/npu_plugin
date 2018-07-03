#ifndef MV_JSON_VALUE_HPP_
#define MV_JSON_VALUE_HPP_

#include <string>
#include <map>
#include <memory>
#include "include/mcm/base/json/value_content.hpp"
#include "include/mcm/base/json/exception/value_error.hpp"

namespace mv
{

    namespace json
    {

        class Object;

        enum class JSONType
        {
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
            std::unique_ptr<ValueContent> content_;

        public:

            Value();
            Value(float value);
            Value(int value);
            Value(const std::string& value);
            Value(bool value);
            Value(const Object& value);
            Value(const Array& value);
            Value(const Value& other);
            Value& operator=(float value);
            Value& operator=(int value);
            Value& operator=(const std::string& value);
            Value& operator=(bool value);
            Value& operator=(const Object& value);
            Value& operator=(const Array& value);
            Value& operator=(const Value& other);
            Value& operator[](const std::string& key);
            Value& operator[](unsigned idx);
            std::string stringify() const;
            JSONType valueType() const;
            template <class T_value>
            T_value& get()
            {
                
                return (T_value&)(*content_);

            }

            
        };  

    }

}

#endif // MV_JSON_VALUE_HPP_