#ifndef MV_JSON_BOOL_HPP_
#define MV_JSON_BOOL_HPP_

#include "include/mcm/base/json/value_content.hpp"

namespace mv
{

    namespace json
    {

        class Bool : public detail::ValueContent
        {

            bool value_;

        public:

            Bool(bool value);
            explicit operator bool&() override;
            std::string stringify() const override;

        }; 
        
    }

}

#endif // MV_JSON_BOOL_HPP_