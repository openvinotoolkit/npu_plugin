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

            double value_;

        public:

            NumberFloat(double value);
            explicit operator double&() override;
            std::string stringify() const override;
            std::string stringifyPretty() const override;
            bool operator==(const NumberFloat& other) const;
            bool operator!=(const NumberFloat& other) const;

            virtual std::string getLogID() const override;

        };  

    }

}

#endif // MV_JSON_NUMBER_FLOAT_HPP_