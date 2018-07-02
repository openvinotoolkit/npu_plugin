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

            String(Object& owner, const std::string& key, const std::string& value);
            explicit operator std::string&() override;
            std::string stringify() const override;

        }; 
        
    }

}

#endif // MV_JSON_STRING_HPP_