#ifndef MV_JSON_NUMBER_INTEGER_HPP_
#define MV_JSON_NUMBER_INTEGER_HPP_

#include <sstream>
#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class NumberInteger : public Value
        {

            int value_;

        public:

            NumberInteger(Object& owner, const std::string& key, int value);
            explicit operator int&() override;
            std::string stringify() const override;

        };  

    }

}

#endif // MV_JSON_NUMBER_INTEGER_HPP_