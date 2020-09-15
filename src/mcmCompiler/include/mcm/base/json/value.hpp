#ifndef MV_JSON_VALUE_HPP_
#define MV_JSON_VALUE_HPP_

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <type_traits>
#include "include/mcm/base/json/value_content.hpp"
#include "include/mcm/base/exception/value_error.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/logger/log_sender.hpp"

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
    
        class Value : public LogSender
        {

            static const std::map<JSONType, std::string> typeString_;
            JSONType valueType_;
            std::unique_ptr<detail::ValueContent> content_;

        public:

            Value();
            Value(double value);
            Value(long long value);
            Value(const char * value);
            Value(const std::string& value);
            Value(bool value);
            Value(const Object& value);
            Value(const Array& value);
            Value(const Value& other);

            virtual ~Value();

            Value& operator=(double value);
            Value& operator=(long long value);
            Value& operator=(const std::string& value);
            Value& operator=(bool value);
            Value& operator=(const Object& value);
            Value& operator=(const Array& value);
            Value& operator=(const Value& other);

            Value& operator[](const std::string& key);
            const Value& operator[](const std::string& key) const;
            Value& operator[](std::size_t idx);
            const Value& operator[](std::size_t idx) const;

            bool operator==(const Value& other) const;
            bool operator!=(const Value& other) const;

            Value& last();
            bool hasKey(const std::string& key) const;
            std::vector<std::string> getKeys() const;
            void emplace(const std::pair<std::string, Value>& member);
            void append(const Value& element);
            std::size_t size() const;

            std::string stringify() const;
            std::string stringifyPretty() const;

            JSONType valueType() const;
            static std::string typeName(JSONType typeID);

            std::string getLogID() const;

            template <class T_value>
            T_value& get()
            {
                return const_cast<T_value&>(static_cast<const Value*>(this)->get<T_value>());
            }

            template <class T_value>
            const T_value& get() const
            {
                switch (valueType_)
                {

                    case JSONType::Object:
                        if (!std::is_same<T_value, json::Object>::value)
                            throw(ValueError(*this, "Unable to return non json::Object value from a " + typeName(JSONType::Object) +
                                " content"));
                        break;

                    case JSONType::Array:
                        if (!std::is_same<T_value, json::Array>::value)
                            throw(ValueError(*this, "Unable to return non json::Array value from a " + typeName(JSONType::Array) +
                                " content"));
                        break;

                    case JSONType::String:
                        if (!std::is_same<T_value, std::string>::value)
                            throw(ValueError(*this, "Unable to return non std::string value from a " + typeName(JSONType::String) +
                                " content"));
                        break;

                    case JSONType::NumberInteger:
                        if (!std::is_same<T_value, long long>::value)
                            throw(ValueError(*this, "Unable to return non long long value from a " + typeName(JSONType::NumberInteger) +
                                " content"));
                        break;

                    case JSONType::NumberFloat:
                        if (!std::is_same<T_value, double>::value)
                            throw(ValueError(*this, "Unable to return non double value from a " + typeName(JSONType::NumberFloat) +
                                " content"));
                        break;

                    case JSONType::Bool:
                        if (!std::is_same<T_value, bool>::value)
                            throw(ValueError(*this, "Unable to return non bool value from a " + typeName(JSONType::Bool) +
                                " content"));
                        break;

                    case JSONType::Null:
                        throw(ValueError(*this, "Unable to return value from a " + typeName(JSONType::Null) +
                                " content"));
                        break;

                }
                return static_cast<T_value&>(*content_);
            }

            
        };  

    }

}

#endif // MV_JSON_VALUE_HPP_
