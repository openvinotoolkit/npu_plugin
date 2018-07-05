#ifndef MV_JSON_NUMBER_FLOAT_HPP_
#define MV_JSON_NUMBER_FLOAT_HPP_

#include <sstream>
#include "include/mcm/base/json/value_content.hpp"

namespace mv
{

    namespace json
    {

        class NumberFloat : public detail::ValueContent
        {

            float value_;

        public:

            NumberFloat(float value);
            explicit operator float&();
            std::string stringify() const override;

        };  

    }

}

#endif // MV_JSON_NUMBER_FLOAT_HPP_