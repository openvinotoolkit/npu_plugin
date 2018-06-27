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

            NumberInteger(int value);
            operator int() const;
            
        };  

    }

}

#endif // MV_JSON_NUMBER_INTEGER_HPP_