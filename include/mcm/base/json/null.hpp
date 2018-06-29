#ifndef MV_JSON_NULL_HPP_
#define MV_JSON_NULL_HPP_

#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class Null : public Value
        {

        public:

            Null() :
            Value(JSONType::Null)
            {

            }
            
        };  

    }

}

#endif // MV_JSON_NUMBER_INTEGER_HPP_