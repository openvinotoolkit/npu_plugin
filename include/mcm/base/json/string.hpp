#ifndef MV_JSON_STRING_HPP_
#define MV_JSON_STRING_HPP_

#include "include/mcm/base/json/value_content.hpp"

namespace mv
{

    namespace json
    {

        class String : public detail::ValueContent
        {

            std::string value_;

        public:

            String(const std::string& value);
            explicit operator std::string&() override;
            std::string stringify() const override;

        }; 
        
    }

}

#endif // MV_JSON_STRING_HPP_