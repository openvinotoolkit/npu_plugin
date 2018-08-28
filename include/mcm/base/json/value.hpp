#ifndef MV_JSON_VALUE_HPP_
#define MV_JSON_VALUE_HPP_

#include <string>
#include <map>
#include <vector>
#include <memory>
#include "include/mcm/base/json/value_content.hpp"
#include "include/mcm/base/exception/value_error.hpp"

namespace mv
{

    namespace json
    {

        class Object;
        class Array;

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
            std::unique_ptr<detail::ValueContent> content_;

        public:

            Value();
            Value(double value);
            Value(int value);
            Value(unsigned int value);
            Value(const char * value);
            Value(long long value);
            Value(const std::string& value);
            Value(bool value);
            Value(const Object& value);
            Value(const Array& value);
            Value(const Value& other);
            Value& operator=(double value);
            Value& operator=(long long value);
            Value& operator=(const std::string& value);
            Value& operator=(bool value);
            Value& operator=(const Object& value);
            Value& operator=(const Array& value);
            Value& operator=(const Value& other);
            Value& operator[](const std::string& key);
            const Value& operator[](const std::string& key) const;
            Value& operator[](unsigned idx);
            Value& last();
            bool hasKey(const std::string& key) const;
            std::vector<std::string> getKeys() const;
            void append(const std::pair<std::string, Value>& member);
            void append(const Value& element);
            unsigned size() const;
            std::string stringify() const;
            std::string stringifyPretty() const;
            JSONType valueType() const;
            template <class T_value>
            T_value& get()
            {
                
                return (T_value&)(*content_);

            }

            template <class T_value>
            const T_value& get() const
            {
                
                return (T_value&)(*content_);

            }

            
        };  

    }

}

#endif // MV_JSON_VALUE_HPP_
