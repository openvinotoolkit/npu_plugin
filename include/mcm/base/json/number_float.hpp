#ifndef MV_JSON_NUMBER_FLOAT_HPP_
#define MV_JSON_NUMBER_FLOAT_HPP_

#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class NumberFloat : public Value
        {

            float value_;

        public:

            NumberFloat(float value) :
            Value(JSONType::NumberFloat),
            value_(value)
            {

            }

            operator float() const
            {
                return value_;
            }
            
            float& get()
            {
                return value_;
            }

            void set(float value)
            {
                value_ = value;
            }

            operator float&() override
            {
                return value_;
            }
            
        };  

    }

}

#endif // MV_JSON_NUMBER_FLOAT_HPP_