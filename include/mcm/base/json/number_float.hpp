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

            NumberFloat(float value);
            operator float() const;
            
        };  

    }

}

#endif // MV_JSON_NUMBER_FLOAT_HPP_