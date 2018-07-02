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

            Null(Object& owner, const std::string& key);
            std::string stringify() const override;

        };  

    }

}

#endif // MV_JSON_NUMBER_INTEGER_HPP_