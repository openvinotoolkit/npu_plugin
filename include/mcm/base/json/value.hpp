#ifndef MV_JSON_VALUE_HPP_
#define MV_JSON_VALUE_HPP_

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
    
        class Value
        {

            JSONType valueType_;

        public:

            Value(JSONType valueType) :
            valueType_(valueType)
            {

            }

            virtual ~Value()
            {
                
            }

            template <class T_value>
            T_value& get()
            {
                
                return static_cast<T_value&>(*this);

            }

            virtual operator float&()
            {
                throw std::logic_error("");
            }

            virtual operator int&()
            {
                throw std::logic_error("");
            }

            virtual operator std::string&()
            {
                throw std::logic_error("");
            }

        };  

    }

}

#endif // MV_JSON_VALUE_HPP_