#ifndef MV_JSON_STRING_HPP_
#define MV_JSON_STRING_HPP_

#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class String : public Value
        {

            std::string value_;

        public:

            String(const std::string& value);
            operator std::string() const;
            
        };  

    }

}

#endif // MV_JSON_STRING_HPP_