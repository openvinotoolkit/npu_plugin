#ifndef MV_JSON_NULL_HPP_
#define MV_JSON_NULL_HPP_

#include "include/mcm/base/json/value_content.hpp"

namespace mv
{

    namespace json
    {

        class Null : public detail::ValueContent
        {

        public:

            Null();
            std::string stringify() const override;
            std::string stringifyPretty() const override;

            virtual std::string getLogID() const override;
            
        };  

    }

}

#endif // MV_JSON_NUMBER_INTEGER_HPP_