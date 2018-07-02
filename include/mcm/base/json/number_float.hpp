#ifndef MV_JSON_NUMBER_FLOAT_HPP_
#define MV_JSON_NUMBER_FLOAT_HPP_

#include <sstream>
#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class NumberFloat : public Value
        {

            float value_;

        public:

            NumberFloat(Object& owner, const std::string& key, float value);
            explicit operator float&() override;
            std::string stringify() const override;

        };  

    }

}

#endif // MV_JSON_NUMBER_FLOAT_HPP_