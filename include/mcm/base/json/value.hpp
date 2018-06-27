#ifndef JSON_VALUE_HPP_
#define JSON_VALUE_HPP_

#include <string>
#include <stdexcept>

namespace mv
{

    namespace json
    {

        class NumberInteger;
        class NumberFloat;
        class String;

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

        template <class T_value>
        class Value
        {

            //JSONType valueType_;

        public:

            Value()
            {

            }

            //JSONType getType() const;

            T_value get()
            {
            
                return *static_cast<T_value*>(this);

            }

            template <class T_assign>
            bool set(const T_assign& value)
            {
                switch(getType())
                {
                    
                    case JSONType::NumberInteger:
                    {
                        if (std::is_integral<T_value>::value)
                        auto ptr = static_cast<NumberInteger*>(this);
                        ptr->
                    }

                    case JSONType::NumberFloat:
                        return *static_cast<NumberFloat*>(this);

                    case JSONType::String:
                        return *static_cast<String*>(this);

                    default:
                        throw std::logic_error("Unknown value type");

                }

            }

        };  

    }

}

#endif // JSON_VALUE_HPP_