#ifndef MV_JSON_NUMBER_INTEGER_HPP_
#define MV_JSON_NUMBER_INTEGER_HPP_

#include <sstream>
#include "include/mcm/base/json/value_content.hpp"

namespace mv
{

    namespace json
    {

        class NumberInteger : public detail::ValueContent
        {

            long long value_;

        public:
    
            NumberInteger(long long value);
            explicit operator long long&() override;
            std::string stringify() const override;
            std::string stringifyPretty() const override;
            bool operator==(const NumberInteger& other) const;
            bool operator!=(const NumberInteger& other) const;

            virtual std::string getLogID() const override;

        };  

    }

}

#endif // MV_JSON_NUMBER_INTEGER_HPP_