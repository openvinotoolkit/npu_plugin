#ifndef MV_JSON_NUMBER_INTEGER_HPP_
#define MV_JSON_NUMBER_INTEGER_HPP_

#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class NumberInteger : public Value
        {

            int value_;

        public:

            NumberInteger(int value) :
            Value(JSONType::NumberInteger),
            value_(value)
            {

            }

            operator int() const
            {
                return value_;
            }
            
            int& get()
            {
                return value_;
            }

            void set(int value)
            {
                value_ = value;
            }
            
        };  

    }

}

#endif // MV_JSON_NUMBER_INTEGER_HPP_